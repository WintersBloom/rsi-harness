# PyTRiO 官方模型接入实测（2026-11-17）

用赛题群发的 Key 直接打了官方端点，确认了三件事：模型名对不对、能不能关思考、每轮要花多少 token 和时间。

> **Key 不进仓库、不进镜像、不进本文件。** 由环境变量 `PYTRIO_API_KEY` 注入，本仓库不含任何密钥；平台提交时填在表单 `api_key` 字段，由 host gateway 代持。企业赛结束后建议在控制台吊销重发。

---

## 1. 端点与模型名（实测）

```python
BASE = "https://pytrio.com/api/openai/v1"     # OpenAI 兼容
KEY  = os.environ["PYTRIO_API_KEY"]            # 环境变量注入，代码与文档里都不落库
```

`GET /models` 返回（0.08 s）：

```
["Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-4B"]
```

**与平台表单 `allowed_models` 完全一致**（`Qwen/Qwen3.8-27B`、`Qwen/Qwen3.5-4B`）。所以表单里 `model` 填 `Qwen/Qwen3.8-27B` 是**字面事实**，不是猜测——我们确实会调用它。

注意模型名带 `Qwen/` 前缀，我们代码里默认的 `qwen3.8-27b` 是校内端点的写法，两者不通用。

## 2. 这是个"先思考再回答"的模型，而且关不掉

PyTRiO 不支持 `chat_template_kwargs`：带 `{"enable_thinking": false}` 直接 **HTTP 400**。校内 vLLM 端点支持这个参数，所以这条差异只在官方端点上出现。

后果是思考会吃掉整个输出预算，`content` 直接为空（`finish_reason=length`）：

| `max_tokens` | 耗时 | completion | reasoning | `content` |
|---|---|---|---|---|
| 200 | 5.0 s | 200 | 200 | **空** |
| 400 | 9.7 s | 400 | 400 | **空** |
| 700 | 12.3 s | 512 | 470 | `{"action":"retrain_10","reason":"…"}` |
| 1000 | 26.2 s | 1000 | 1000 | **空**（思考失控） |

⇒ **只给 200 token 时，模型 100% 不会给出答案**。我们 `ask_model` 原来写的是 `max_tokens=min(200, rnd.max_output_tokens)`，在官方模型上等于"每轮都退回规则"，而且日志会写成"模型给了空字符串"——**看起来像模型偶尔抽风，实际是预算不够**。

## 3. 有效的三件事

1. **预算给到 900**（思考约 330–670 token，留余量）；
2. **在 user 消息末尾加 `/no_think`**——Qwen3 的约定，实测把思考压到 330 token 左右；
3. **空回复重试一次**（上面第 4 行那种思考失控是小概率但会发生）。

修完后连打 6 次（同一输入）：

```
OK  8.89 s completion=364 reasoning=334
OK  8.81 s    ...        ...        ...
6/6 成功，completion 恒为 364（temperature=0 下确定性）
```

**每轮约 9 秒、约 364 token**。10 轮 ≈ 90 秒、3.6k token，对 `token_budget=8000000` 和 `wall_seconds=21600` 都是零头。

## 4. `Qwen/Qwen3.5-4B` 不适合做这件事

实测 **58.7 秒** 才出结果，同样是思考模型、同样空 `content`。拿它做每轮决策会拖到评测超时，只用它做对照实验。

## 5. 已改的代码（`src/contest_loop.py::ask_model`，同步到 `harness_sdk/app/`）

```python
cap = int(getattr(rnd, "max_output_tokens", 0) or 0)
budget = max(1, min(900, cap)) if cap > 0 else 900     # 绝不超平台上限，也不虚抬下限
for attempt in range(2):                               # 空回复重试一次
    receipt = rnd.model([{"role": "user", "content": prompt + "\n/no_think"}],
                        max_tokens=budget, temperature=0)
    last_text = (receipt.text or "").strip()
    if last_text:
        break
```

三条约束：

1. **绝不超 `rnd.max_output_tokens`**——SDK `round.py:179` 规定超了直接抛 `MAX_TOKENS_INVALID`，整轮作废。所以不能用 `max(200, …)` 那种写法把下限抬起来。
2. **异常与空回复都走 `ask_model` 的返回值第四个分量 `llm_status`**，取值 `answered` / `empty` / `out_of_set` / `error`（B 包是 `b_ack` / `b_empty` / `b_error`）。
3. **`change_summary` 里写死 `max_out=<实际预算>；llm_status=<状态>`**，和 `parent_found`/`trained`/`before`/`after` 并列——第一次试运行就能从平台回传的记录里读到"平台到底给了多少 token 上限、模型到底出没出内容"，不用额外消耗 trial。

## 6. 🔴 未解决：平台侧的 `max_output_tokens` 可能是 128

**这一条没有解决，必须先说清楚。**

PyTRiO 端点上思考要 **330–670 token**（`/no_think` 压到 334），所以：

| 平台给的上限 | 结果 |
|---|---|
| ≥ 700 | 正常：模型出 JSON，动作由模型选 |
| 200 | 模型调用成功、`call_id` 有、`content` 为空 ⇒ 决策退回规则 |
| **128** | **同上，永远退回规则** |

而 SDK 自带的测试与模板给的是 **128** 这个量级：

```
tests/test_sdk.py:58                       "max_output_tokens": 128
tests/test_transport_regressions.py:11     max_output_tokens = 128
template/contest_submission.py:19          max_tokens=min(32, rnd.max_output_tokens)
```

官方模板自己写 `min(32, …)` —— 如果平台给的是 700+，模板没必要压到 32。

**所以 sdk4 里我们做的是"把不确定性变成可观测"，不是"解决它"**：真到了 128，模型这一步就出不了答案，调用成立（`model_call_ids` 非空）、决策回退规则层。这是**平台预算与模型形态的匹配问题**，不是实现缺陷——`change_summary` 里的 `max_out=` 会给出一手证据。

## 7. 其他还没验证的

- `rnd.model()` 走的 host gateway 是否透传 `/no_think`（大概率透传，同一条 OpenAI 兼容协议）；
- 平台侧是否对单次调用有超时（9 秒应该安全）；
- 平台是否允许同一轮多次调用（当前实现空回复会重试一次，即 1–2 次）。

## 8. 两条通用教训

- **本地跑通的模型行为不能代表平台跑通**：校内 vLLM 支持 `chat_template_kwargs` 关思考、官方端点 400。和"本地 numpy 1.23.5 与镜像 1.26.4 给出不同末位"是同一条。
- **`Qwen/Qwen3.5-4B` 不可用于每轮决策**：单次 58.7 秒（同一 prompt、`max_tokens=900`），10 轮将达 10 分钟，超出训练轮时间预算。

## 9. 安全

Key 只经环境变量 `PYTRIO_API_KEY` 注入，本仓库不含任何密钥（含本文档）。企业赛结束后在 PyTRiO 控制台**吊销重发**；在此之前，截图、录屏、提交材料里都不要出现。
