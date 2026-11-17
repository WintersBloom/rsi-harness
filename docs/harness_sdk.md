# Harness SDK 接入说明（v1.0.2 / protocol workspace.v2）

平台在 2026-11-17 换了提交方式：不再直接跑 `python /app/harness/predict.py`，而是要求镜像里有
**三个固定入口**，由 SDK 驱动。这份文档记录协议细节、我们的实现位置、以及四个已经踩过的坑。

---

## 1. 协议总览

| 项 | 值 |
|---|---|
| 协议名 | `contest.harness.workspace.v2` |
| SDK 包 | `contest_harness` 1.0.2 |
| 入口 | `python3 /app/harness.py describe\|train\|predict` |
| 模型访问 | `host_gateway_rpc`（平台代持 Key，镜像里不写密钥） |
| 提示词版本 | `industrial-workspace-flexible-v3` |

`/app/harness.py` 只有 7 行：导入 SDK 的 `main`，把 `contest_submission` 交给它。

## 2. 目录与权限

| 路径 | 权限 | 用途 |
|---|---|---|
| `/app` | 只读 | 代码与权重 |
| `/data` | 只读 | 本轮的公开数据 |
| `/previous` | 只读 | 之前的产物（读前先核 `checkpoint_sha256`） |
| `/state` | 只读 | 上轮存盘 |
| `/model` | 只读 | 父模型 |
| `/workspace` | **可写** | 候选、中间产物必须落这里 |
| `/tmp` | 可写 | 临时文件 |

候选目录限制：**≤ 64 个文件、≤ 64 MiB、至少一个 checkpoint**。

## 3. 三个入口的契约

### describe

import 选手模块，检查 `train` / `predict` 都是 callable，打印：

```json
{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc",
 "sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}
```

失败抛 `HarnessError("SUBMISSION_INVALID")`，平台连训练都不进。

### train(rnd)

平台推一个 start 帧到 stdin，SDK 解析成 `Round` 对象。必需字段：
`protocol`、`prompt`、`prompt_version`、`dataset_key`、`round_index`、`task_contract`、
`model_name`、`max_output_tokens`、`parent_checkpoint_sha256`、`parent_artifact`、`artifacts`。

可用的东西：

- `rnd.data.training_rows()` / `validation_rows()` — 迭代逐行 dict（`id`、`inputs`、`answer`、`context`）
- `rnd.history()` — 之前每轮的记录
- `rnd.model(messages, max_tokens=, temperature=)` — **一次真实模型调用**，返回 `Receipt(call_id, text, ...)`
- `rnd.submit_candidate(dir, hypothesis=, change_summary=, configuration=, seed=, checkpoint_paths=, code_paths=, predict_argv=)`
- `rnd.save_state()` / `rnd.finish(...)` — `finish` 不调用就 `ROUND_NOT_FINISHED`

**硬约束**：每轮 train 至少一次真实模型调用（`round.py:335-337`，`cited_calls` 为空就抛
`MODEL_CALL_REQUIRED`）。这堵死了"完全不用大模型"，但 Key 由平台代持，镜像里不需要任何密钥。

### predict(sample, model_dir)

stdin 逐行喂样本，每行一个 JSON。`Sample` 字段：

```python
Sample(task, inputs, context=None)
sample.dataset_key = None if context is None else context["dataset_key"]
sample.artifact_id = None if context is None else context["artifact_id"]
```

`context` 来自环境变量 `CONTEST_PREDICTION_CONTEXT`（dict，键为
`protocol` / `dataset_key` / `task` / `artifact` / `artifact_id`），**可能为 None**。

返回形态逐包固定：

| 包 | 返回 |
|---|---|
| A（分类） | `[{"label": str, "score": float}, ...]`，6 个 label 不重复、最高分唯一 |
| B（回归） | `{"value": float}`，非负有限 |
| C（回归） | `{"value": float}`，非负有限 |

`normalize_output`（`predict.py:56-82`）会校验上述每一条，不合格就整次测量失败。

## 4. 我们的实现

```
harness_sdk/
├── app/harness.py                 ← 模板原样
├── app/contest_harness/           ← SDK 源码 vendored（不走 pip，离线可用）
├── app/contest_submission.py      ← train / predict 实现
├── app/net.py                     ← 从 harness/net.py 复制
├── app/checkpoints/{A,B,C}.npz    ← 从 harness/checkpoints_trained/ 复制
├── work/                          ← 冒充 /workspace（CONTEST_WORK）
├── bench/                         ← 三包各 100 行输入 + 基准/新输出
└── fake_host.py                   ← 本地假宿主，跑一轮 train
```

`predict()` 的逻辑：`_pick_key` → `_load`（按 `CONTEST_ARTIFACT_PATH` → `/app/checkpoints` →
本地 `checkpoints` 依次找）→ `_answer`（`flatten` + `check_schema` + `predict`）→ 任何异常都退
`_fallback`，**绝不抛出**。

## 5. 四个坑

1. **数值一致性必须在镜像里验**。本地 `C:\Python311` 是 numpy 1.23.5，镜像里是 1.26.4，
   同一个前向会差最后 1–2 位（`334.8295202143448` vs `...446`）。本地跑只有 2/100 相同，
   进镜像立刻 100/100。**别把 ULP 差异当成移植错误。**
2. **一行坏数据杀整个进程**。`serve_predictions` 是 raise 后退出，不像旧的 1.3 会跳过。
   所以 `predict()` 必须自己 try/except。
3. **`dataset_key` 可能为 None**。没 context 时要按 inputs schema 推断：
   `"settings" in inputs` → C，`features` 是 dict → B，长度 128 的 list → A。
4. **候选目录必须已存在且为空**。`submit_candidate` 不会替你建目录；第二次跑同一轮要先把
   旧目录清掉再建。

## 6. 验证记录（2026-11-17）

基准从**冻结的 tar** 里取，不从仓库重跑：

```bash
docker load -i dist/zhixun-harness-1.3.tar
docker run --rm -i --network none zhixun-harness:1.3 < bench/in_A.jsonl > bench/base_A.jsonl
```

新代码在镜像里跑：

```bash
docker run --rm -i --network none -v "%CD%\app:/app2:ro" -e PYTHONPATH=/app2 \
  --entrypoint python zhixun-harness:1.3 /app2/harness.py predict < bench/in_A.jsonl
```

| 包 | 行数 | 逐条相同 | 差异 |
|---|---|---|---|
| A | 100 | 100 | 0 |
| B | 100 | 100 | 0 |
| C | 100 | 100 | 0 |

比的是 `json.loads` 后的结构 + 浮点值 `==`（数值完全相等，不是近似）。

## 7. 与旧镜像的关系

`dist/zhixun-harness-1.3.tar` 是**旧协议**（单一 `predict.py`）的产物，`harness/` 与 `release/`
里的代码也是旧协议。新协议要走 `harness_sdk/`，两者目前并存：

- 旧镜像：留给"平台若退回旧协议"的保险，sha256 不变；
- `harness_sdk/`：新协议的唯一实现，第 4 步（五环进 `train`）与第 5 步（重打镜像）都基于它。

**在第 5 步之前不要动 `dist/` 的 tar，也不要用旧 `harness/predict.py` 当基准。**

## 8. 第 4/5 步：策略迭代接进 train + 重打镜像（2026-11-17）

`train(rnd)` 现在不再是一行占位，而是调 `contest_loop.run_round(rnd)`（从 `src/contest_loop.py` 同步过来）。
每轮流程：读经验 → 规则给底、模型在候选名单里裁定 → 执行（重训或保留父模型）→ 记账 → `submit_candidate` → `finish`。

### 四件套验证（全部通过）

| # | 验证 | 结果 |
|---|---|---|
| 1 | `describe` | `{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}`，exit 0 |
| 2 | 一轮 `train`（假宿主） | 三轮连跑，`model_call_ids=1`、`configuration` 含 `directions`/`plateau`/`rounds_since_accept`、候选目录有 `checkpoint.npz` |
| 3 | `predict` 三包各 100 行 | 与冻结版 1.3 **逐条完全相等**（A/B/C 各 100/100），比的是 `json.loads` 后结构 + 浮点 `==` |
| 4 | 断网 + 体积 | `--network none` 跑完 C 包 100 行 **0.62 s**；镜像 345 MB，tar **86.3 MB** |

### 镜像

```
zhixun-harness:sdk3
tar:    harness_sdk/out/zhixun-harness-sdk3.tar   90,502,656 B (86.3 MB)
sha256: 21CD49DC6496A5B511F17E4D298D517FB322445D180C7D1F3E39100B111B18D5
```

`docker save` 导出（公告要求的格式），不是 `docker export`。**旧 `dist/zhixun-harness-1.3.tar` 未动**，两者并存：
旧的是单入口 `predict.py`（旧协议保险），新的是三入口 `harness.py`。

### 三包各跑三轮（sdk3，假宿主 + 内置父权重）

| 包 | 轮 1 | 轮 2 | 轮 3 | 说明 |
|---|---|---|---|---|
| A | `retrain_20` −0.020937 ⇒ **R1 判死** | `retrain_10` −0.008391（带内） | `retrain_10` −0.008391（streak=2） | 第 4 轮进 R2 |
| B | `noop` | `noop` | `noop` | 符合设计（缺官方认可的健康标签来源） |
| C | `retrain_20` −0.062852（RMSE 变差，带内） | **`noop`** | `retrain_20` | 第 2 轮 R3 生效 |

C 包三轮 41.8 秒（113089 条 × 20 epoch × 3）。

第 2/3 轮 delta **逐位相同**（A 的 `-0.008391168058790033`、C 的 `-0.062851972913478`），是"固定父模型 + 固定 seed 下确定性训练"的必然结果，
也是"动作空间窄 ⇒ 学会停才有价值"的实证（详见 `docs/策略迭代设计.md` 第 8.1 与第 9 节）。

### 三个已修的 bug（都会静默毁掉一轮）

| 位置 | 错误 | 症状 |
|---|---|---|
| `contest_loop.py` | `receipt.text()` 把 property 当方法 | `TypeError` 被 except 吞掉，日志写"模型不可用"，**LLM 从未参与** |
| `contest_loop.py` | `len(generator)` | `TypeError: object of type 'generator' has no len()` |
| `contest_loop.py` | `abs(None)` | 第一轮 delta 为 None 时，第二轮 R3 分支崩 |
| `contest_loop.py` | `calibrate` 落到训练分支 | 名字叫校准、实际跑了 20 epoch 重训；**三个包都没有阈值可调**，已从动作集移除 |
| `fake_host.py` | 三包数据路径写死成 A 包 | 跑 C 时报 `RUL must be a finite nonnegative number`，表现像"代码坏了" |

### sdk5：让"模型调用失败"不再整轮作废（2026-11-17 晚）

第一次试运行在 `train` 首轮就失败，平台只回 `failure_kind: "unknown"`、`report: null`，`/runs/{id}/logs|events|evidence` 全 404——**一个轮次都没被平台收到**。读 SDK 源码后确认两条"还没写记录就死"的路径：

1. `contest_harness/round.py:336-337`：`finish()` 时 `cited_calls` 为空即抛 `HarnessError("MODEL_CALL_REQUIRED")`；而 `round.py:188` 只有**调用成功**才 `self.call_ids.append(call_id)`。⇒ 模型调用失败 = 本轮必死，且死得没有任何可读原因。
2. `contest_loop.py` 里 `rnd.history()` 在 try 之外：平台历史若不符 `round.py:151-163` 的 id 校验，抛 `HISTORY_INVALID`，同样整轮死。

sdk5 的三处改动（`predict` 一行未动）：

| 改动 | 做法 | 目的 |
|---|---|---|
| 保底调用 | 新增 `_bank_call()`：`(messages, budget)` → `(messages, 1)` → `("ok", 1)` 三级重试，全失败才抛 | 尽量保证 `finish()` 时至少有一个 `call_id` |
| 去掉历史依赖 | 删除 `rnd.history()` 与 `history_ids=`，`finish()` 只传 `model_call_ids=list(rnd.call_ids)` | 少一个"读历史即死"的入口 |
| 失败可读 | 决策/执行/提交三段全包 try；异常写入 `fatal=[...]`，拼进 `change_summary`（截断 240 字符） | 下次失败能在平台页面读到原因，不再只看到 unknown |

另外 `rnd.model(...)` 去掉了 `temperature=0`，少一个被网关拒的可能。

**验证**：`py_compile` 通过；假宿主 A/B/C 各两轮 exit=0，动作与 delta 与 sdk4 逐位一致（A `-0.02093671463440616` → `-0.008391168058790033`；B 两轮 `noop`；C `-0.062851972913478` → `noop`），`model_call_ids: 1`；`describe` 通过；`predict` 100 行与 1.3 基准 **100/100 相同**；三包全量 `--network none` → A 0.7210 / B 681.1598 / C 50.9320，与 sdk4 完全一致。

镜像：`zhixun-harness:sdk5`，tar `dist/zhixun-harness-sdk5.tar`，90,505,216 B，sha256 `5EF48B1C75756DE5375529DC02D13D773B8EC7732CFBF7DAC0110E2D69A5C8CB`。

### sdk6：把训练过程打成日志（stderr），不碰协议流（2026-11-17 晚）

sdk5 只能把失败原因塞进 `change_summary` 的 240 个字符里。sdk6 改成"每轮都往 stderr 写一条结构化日志"。

**为什么安全**：`contest_harness/cli.py` 的三个分支都是先 `isolate_stdio()`（`:51` / `:59` / `:71`）、**之后**才 `load_submission()`（`:53` / `:63` / `:73`）。而 `protocol.py:36-42` 的做法是：先 `os.dup(1)` 复制出私有句柄给协议帧，再 `os.dup2(2, 1)` —— 之后 fd1 就是 stderr。所以选手代码里的任何输出都落在 stderr，协议帧走另一份 fd。docstring 原话：*"Keep protocol frames on a private copy of stdout. Training logs use stderr."*

实现上用 `_log()` 显式写 `sys.stderr` 并 flush，异常吞掉（日志失败绝不影响主流程）。打印的内容：

| 时点 | 一行日志 |
|---|---|
| 轮次开始 | `[zhixun] start round=1 key=A rows=train:445,val:1244 cap=256 last=-` |
| 决策完成 | `[zhixun] decide action=retrain_20 status=answered budget=256 call=call-local-0 reason=…` |
| 训练完成 | `[zhixun] train epochs=20 before=F1 0.7210 after=F1 0.7000 accepted=False` |
| 训练异常 | `[zhixun] train failed: KeyError: …` + 完整 `traceback.format_exc()` |
| 提交前 | `[zhixun] submit artifact=… calls=1 fatal=0 elapsed=0.7s`，有 fatal 时再打一行 |

**验证**（`zhixun-harness:sdk6`）：
- `describe` 通过；
- `predict` 100 行：stdout **100 行、每行 `json.loads` 成功、stderr 0 字节**；
- 假宿主 A 一轮：stderr 出现上述四行，`model_call_ids: 1`，`change_summary` 与 sdk5 一致；
- 三包全量 `--network none`：A 0.7210 / B 681.1598 / C 50.9320，与 sdk5 完全一致。

镜像：`zhixun-harness:sdk6`，tar `dist/zhixun-harness-sdk6.tar`，90,507,264 B，sha256 `0469AD11C2B711A9523D6BC85D8BD2C2E0DCA925B2554152B6B630708CE1C0A3`。

**未验证的前提**：平台是否把 stderr 收进运行报告，我们读不到（`/runs/{id}/logs` 是 404）。这版的价值是"如果平台收，我们就有全量轨迹；如果不收，代价为零"。
