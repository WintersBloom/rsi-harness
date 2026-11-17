# harness 自测记录（2026-11-14）

## 一、交付了什么

```
harness/
├── predict.py           # stdin 逐行读 → 路由 → 前向 → stdout 逐行写
├── router.py            # 三条 schema 判据（照抄 docs/镜像设计.md）
├── net.py               # 纯 numpy：flatten + 前向（复刻 baselines/model.py）
├── checkpoints/         # 官方原版权重（默认，可复现官方基线）
└── checkpoints_trained/ # 十轮链产出的权重（提交时用这个）
```

`net.py` 不 import `baselines`，前向按官方公式重写：

```
x = clip((x - mean) / scale, -10, 10)
h = tanh(x @ w1 + b1)
z = h @ w2 + b2
分类 → softmax(z)                回归 → max(0, z * target_scale + target_mean)
```

## 二、协议

输入（每行一个 JSON 对象）：

```json
{"task": "classification", "inputs": {...}}
```

输出（同序、逐行、flush）：

```json
{"outputs": [{"label": "Ethanol", "score": 0.8446180407749447}, ...]}   // 六个，按 classes 顺序
{"outputs": {"value": 437.57}}                                          // 回归
```

换权重目录：

```bash
python harness/predict.py --checkpoints harness/checkpoints_trained
```

### 2.1 新协议（SDK 1.0.2）的输入契约：`task` 字段缺了会**整进程死**

新协议的 `contest_harness/predict.py:98` 是硬校验：

```python
if not isinstance(request, dict) or "inputs" not in request or not isinstance(request.get("task"), str):
    raise HarnessError("PREDICTION_REQUEST_INVALID")
```

**请求必须同时带 `task` 和 `inputs`。** 而且 `serve_predictions` 的 for 循环里**没有 try/except 包住这段**——一行不合法 → 整个进程 `exit 1`，这一行之后的所有行**全部丢失**。

实测（执行者踩到）：漏了 `task` 字段，三包全部 `PREDICTION_REQUEST_INVALID`、exit 1、零行输出。

**根因是两个"看起来一样"的格式被当成同一个：**

| 场合 | 每行长什么样 | 出处 |
|---|---|---|
| **数据文件**（官方 `validation.jsonl`） | `{"id": "...", "inputs": {...}, "answer": ...}` | `HARNESS_SDK.md:48` |
| **predict 的 stdin**（平台喂进来的） | `{"task": "...", "inputs": {...}}` | `HARNESS_SDK.md:168`（`:163` 明写"上下文缺失时只使用样本行里的 `task`"） |

⇒ **数据文件格式 ≠ stdin 契约。** 直接 `cat validation.jsonl` 喂 predict 必缺 `task`，报的是契约错，不是模型错。

把验证集转成契约格式、取前 10 行（答辩现场可当场跑）：

```bash
python -c "
import json,sys
for i,line in enumerate(open('data/official/cpu-a-guide-20261110-v1/public/validation.jsonl',encoding='utf-8')):
    if i>=10: break
    o=json.loads(line)
    print(json.dumps({'task':'classification','inputs':o['inputs']}))
" | docker run --rm -i --network none zhixun-harness:sdk4 predict
```

正确格式：

```json
{"task": "classification", "inputs": {"features": [0.1, ...]}}     // A，128 维 list
{"task": "regression",     "inputs": {"features": {"horizontal": {...}, "vertical": {...}}}}  // B
{"task": "regression",     "inputs": {"settings": [...], "sensors": [...]}}                    // C
```

我们能控制的范围只到 `Sample` 之后：**SDK 层的校验失败是契约不匹配，不是我们的实现缺陷**。`predict()` 拿不到原始请求对象，所以这一层兜不了——只能保证"拿到 `Sample` 之后不崩"（`predict()` 整体 try/except → fallback）。

答辩口径（问："你们怎么保证平台喂进来的格式一定对？"）：
> 不保证。这是 SDK 层的硬校验（`predict.py:98`），平台按契约喂就不会触发。我们能控制的是拿到 `Sample` 之后任何异常都不让进程死。

### 2.2 测试基础设施的坑：假宿主必须按 `dataset_key` 选数据

`harness_sdk/fake_host.py` 最初把三包的数据路径都写死成 A 包（`OFFICIAL = ...\cpu-a-guide-20261110-v1\public`）。于是跑 C 包时，`train_run("C", ...)` 拿到的是 A 包的分类标签（字符串），直接抛：

```
ValueError: line 1: RUL must be a finite nonnegative number
```

**危险在于它的表现和"被测代码坏了"几乎一样**：`delta=null`、`rejected`、进程照常 exit 0。区别只有一条——**它只在换包时出现，A 包永远是对的**。

已改为 `PACKAGES = {"A": ..., "B": ..., "C": ...}` 按 key 取路径。

> 测试基础设施出错时的表现，和被测代码出错时的表现很像；**区别是前者只在你换测试对象时出现**。所以换包跑出来的第一个异常，先怀疑宿主，再怀疑业务代码。

### 2.3 sdk4 三入口实跑（2026-11-17，执行者亲跑）

平台发布 Harness SDK 1.0.2 后，`zhixun-harness:1.3` 那版（`/app` 下只有 `harness/`，无 `harness.py`、无 `contest_harness`）**必然过不了 `describe`**，所以重打成 `zhixun-harness:sdk4`。以下是本机实测，全部 `--network none`。

**tar**：`dist/zhixun-harness-sdk4.tar`，90,503,680 字节，sha256 `B63F1B0D6DBBA62502E0B81AB798134E4A7074D0FEAE5E438EF91E826FD59EEE`。

**① `describe`（平台进训练轮的第一个入口）**

```bash
docker run --rm zhixun-harness:sdk4 describe
```

```
{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}
```

exit 0。**这一条不过，平台连训练都不进**（`cli.py:17-31` 会 `HarnessError("SUBMISSION_INVALID")`）。

**② `predict` 三包各 10 行，与 `1.3` 逐位一致**

```bash
# A（分类，inputs.features 是 128 维 list）
python -c "import json;[print(json.dumps({'task':'classification','inputs':json.loads(l)['inputs']})) for l in open('data/official/cpu-a-guide-20261110-v1/public/validation.jsonl',encoding='utf-8').readlines()[:10]]" | docker run --rm -i --network none zhixun-harness:sdk4 predict
```

首行：
```json
{"outputs":[{"label":"Ethanol","score":0.9532940049746401},...共 6 类...]}
```

B 首行 `{"outputs":{"value":437.5695765327886}}`、C 首行 `{"outputs":{"value":182.8339256538419}}`——**三个数与 `zhixun-harness:1.3` 的对应输出逐位相同**（数值 `==`，只差 JSON 空格，因为 SDK 的 dump 与 1.3 的 `json.dumps` 不同）。输入 10 行 → 输出 10 行，无丢行。

**③ 与 1.3 的差异只在 train 侧**

| | 1.3 | sdk4 |
|---|---|---|
| 入口 | 只有 `predict` | `describe` / `train` / `predict` |
| `/app/harness.py` | 无 | 有（官方 7 行模板） |
| `contest_harness` 包 | 无 | 有（vendored，不走 pip） |
| `predict` 结果 | 基准 | **逐位一致** |
| `train(rnd)` | 无 | 策略闭环（`contest_loop.py`） |

**这解决了一个具体的死法**：1.3 交上去 `describe` 失败 ⇒ `SUBMISSION_INVALID` ⇒ 零分，且看不出是"没实现"还是"实现错了"。

## 三、为什么推理期什么都不做（主动选择，不是没做）

初版设计里有五环（监测→归因→决策→执行→回滚），**在镜像里主动关掉了**。三条证据说明平台在结构上不允许运行时适应：

1. `baselines/predict.py:1` 原文："JSONL prediction adapter. Reads inputs, never labels; **no model execution hooks**." —— 平台不提供任何让模型在运行时被改写的钩子。
2. `tests/test_baselines.py:65-72`：输入里只要出现 `RUL` / `true_rul` / `Normalized_RUL` / `TotalSnapshots`，直接 `assertRaises(ValueError)`。⇒ 结构上禁止读标签，连"事后算一下对错"都不行。
3. `TRAINING_GUIDE.md:32`：验证集与隐藏测试集**不得参与拟合任何统计量**。这条比"不许重训"更严：阈值校准、预测分布对齐、标准化重估，全都是"从测试数据里推出一个参数"，**一样违规**。

所以"自进化"不能由运行时动作证明，只能由**提交物**证明：

```
离线：官方包 → 十轮续训链（每轮 --resume）→ 每轮记录假设/代码变更/父检查点/验证指标 → SwanLab
线上：镜像只做「挑一个权重 + 纯前向」，零状态、零拟合
```

这和旧系统（数据窗时代）的结论是同一条：**默认不动，动之前要有依据**；只是到了官方评测，"依据"只能来自离线实验记录。

## 四、两个权重目录

| 目录 | A Macro-F1 | B RMSE | C RMSE | 用途 |
|---|---|---|---|---|
| `checkpoints/`（官方原版） | 0.7140 | 681.1598 | 52.8686 | 复现基线、对照 |
| `checkpoints_trained/`（十轮链） | **0.7210** | 681.1598 | **50.9320** | 提交用 |

- A 取 `out_chain_a/r03`（round 3，+0.70pp）；
- B 取**官方原版**：十轮 6 个动作全负，一个都没被接受 ⇒ 保持不动；
- C 取 `out_chain_c/r07`（round 7，−3.66%）。

`--checkpoints` 默认指向官方目录，所以不传参数时的行为永远可复现基线；提交镜像时改默认或用启动参数指向 trained。

## 五、自测结果（全部实测）

### 5.1 与官方 adapter 逐字节对比

用各包 `public/validation.jsonl` 全量喂入，比对 stdout：

| 包 | 行数 | 官方 rc | 我们 rc | stdout 逐字节相同 | 官方耗时 | 我们耗时 |
|---|---|---|---|---|---|---|
| A | 1,244 | 0 | 0 | **是** | 0.73 s | 0.72 s |
| B | 1,679 | 0 | 0 | **是** | 0.58 s | 0.54 s |
| C | 23,049 | 0 | 0 | **是** | 2.63 s | 2.45 s |

包括浮点数的十进制表示在内，逐字节一致。

### 5.2 分数复现（`--checkpoints` 两组）

| 权重 | A Macro-F1 | B RMSE | C RMSE |
|---|---|---|---|
| 官方原版 | 0.7140（官方 0.7140） | 681.1598 | 52.8686 |
| 十轮链 | **0.7210**（+0.98%） | 681.1598（0%） | **50.9320**（−3.66%） |

A 各类召回（validation，官方权重）：Acetaldehyde **0.000**、Acetone 0.992、Ammonia 1.000、Ethanol 0.963、Ethylene 0.904、Toluene 1.000。

### 5.3 红线验证

| 检查 | 结果 |
|---|---|
| 行里带 `answer` / `meta`，输出是否变化 | **不变**（50 行逐字节相同） |
| 单进程混合流 `A,B,C,A,B,C,A` | 形状 `list,dict,dict,list,dict,dict,list`，**正确**，0.28 s |
| 输入顺序反转，逐条答案是否跟着变 | **不变**（证明无状态拟合） |
| checkpoint 字节未改 | 见下方 SHA256 |

```
官方 A.npz FD1E08B2AAD08B2E9754F9B2DFDC029481C0DA7ECDEDCA0ABB0EC6FCC818C757
官方 B.npz C7D819FCEE9D25459C583B5FFDB35593D4A47B524ED152D0226BF81A283314B4
官方 C.npz 98EC6E30E38AEE56CA15B7D624A8F412226D46FE97B170667A567601C700BC4E
十轮 A.npz E3C9BF73D340FA28C2CFC8B5456617BEDE0ADB14FFF8A34E02A625515CF79C8C
十轮 C.npz 08C0A40F647D780D42CCDA2C739FC4DA5742028D0BCA8017993F46E127E1BDF9
```

### 5.4 性能余量

C 包 23,049 行 2.45 s ⇒ 约 9,400 行/秒。三包 validation 合计 25,972 行约 3 s。评测墙钟 21,600 s，**前向不是瓶颈**，瓶颈在容器启动与镜像拉取。

## 六、三条红线的实现位置

| 红线 | 实现 |
|---|---|
| 不读 `answer` / `meta` | `predict.py` 只取 `request["inputs"]`，其余键不进入任何计算路径 |
| 状态按任务分桶 | `state.setdefault(task, {"n": 0, "labels": {}})`，A/B/C 各自计数（当前只计数，为将来留位） |
| 推理期不做统计量拟合 | `mean/scale` 只在 `Model.__init__` 从 npz 读，`predict()` 是无状态纯函数 |

## 七、复现命令

```powershell
# 逐字节对比 + 红线
python .team\_bytecheck.py

# 两组权重的分数
python .team\_check_trained.py

# 手动跑一条
'{"task":"classification","inputs":{"features":[21749.5635,2.627488, ...]}}' `
  | python harness\predict.py
```

## 八、输入输出时序自检（排除"互等僵死"）

群里报告过一个平台侧现象：「评测通过但过十五秒自动停止」，官方回复「平台侧的问题，在修」。

但存在另一种解释，且如果成立就是**我们自己的问题**：

> 平台在等我们的输出，我们在等平台把 stdin 关掉（EOF）。双方互相等待 → 平台超时。

因为 `predict.py` 是常驻进程、逐行读 stdin，如果它攒到 EOF 才 flush，那么一个"保持 stdin 打开、持续喂数据"的平台调用方会一直等到超时。

### 实测方法（本地，不消耗任何平台配额）

起子进程、**写一行但保持 stdin 打开**，看 3 秒内 stdout 是否有字节。

```powershell
python .tmp\probe_stdin.py
```

### 结果

```
[OK] 逐行应答
     首行延迟 300 ms（stdin 未关闭）
     输出前 80 字符: {"outputs": [{"label": "Ethanol", "score": 0.8446...}]}
结论: PASS
```

**结论**：`predict.py` 在 stdin 未关闭的情况下，300 ms 内就吐出了第一行。它是逐行 flush 的，**不存在"等 EOF"的僵死模式**。

因此平台那个"15 秒停止"如果再次出现，**可以排除我们的设计问题**，归因到平台侧。

### 300 ms 花在哪

不是前向计算（单行 0.1 ms 量级），是进程启动：`import numpy` + 加载三个 checkpoint。这部分在容器里会有冷启动开销，平台如果按"启动即计时"算超时，需要预留。

### 仍需确认的一条

平台调用方式是「逐行喂」还是「一次性把整个文件塞进 stdin 再关掉」。两种情况我们都能正确工作（前者 300 ms 首行、后者跑完全量），但**前者要求平台不因这 300 ms 判定失败**。这条只能问官方，无法本地验证。

---

## 九、stdin 读取策略：最终选"读到 EOF"，与官方一致

### 先看官方怎么写

`baselines/predict.py:30` 就一句：

```python
for line in sys.stdin:
    if not line.strip():
        continue
    ...
    print(json.dumps(predict(model, request["inputs"]), allow_nan=False), flush=True)
```

**平台只要关掉 stdin，官方 baseline 就会正常退出。** 反过来：如果平台"写完不关"，官方自己也跑不完——那是平台的问题，不该由选手承担失败次数。所以"平台一定会关"这个判断，有官方代码背书。

### 我们加过一个 60 秒空闲超时，然后把它默认关掉了

`harness/predict.py` 用 `select` 实现了 `--idle-exit`：连续 N 秒无输入就主动退出，防"写完不关导致永久阻塞"。第一版默认 60 秒。

**推翻它的理由**：如果平台是**分批喂**、批间隔超过 60 秒，我们会在中途自己退出，后面的行全丢——不完整输出比慢输出严重得多，宁可慢。而"写完不关"这个风险已被上面的官方写法排除。所以默认改回 `0`（读到 EOF），与官方同策略；参数留着，万一官方确认是"写完不关"，加 `--idle-exit 60` 即可。

### 实测：两种喂法都跑通（容器内，`--idle-exit 2` 放大验证）

| 喂法 | 首行延迟 | 进程退出 | 退出码 |
|---|---|---|---|
| 流式写一行、**不关** stdin | 1.07 s | 3.33 s（2 s 空闲 + 启动） | 0 |
| 写一行后**关闭** stdin | 0.44 s | 0.66 s（读到 EOF 立即退） | 0 |

两种都正确应答，说明 `select` 分支与 `for line` 分支行为一致。

### 用 1.3 冻结版复测（2026-11-15 17:20，规划者要求"先做这一行测试"）

脚本 `.team/_timeout30.py`：写一行 → 最多等 5 秒 → 再写第二行 → 关 stdin。

| 场景 | 首行延迟 | 第二行 | 退出 |
|---|---|---|---|
| 容器已热（连跑两次） | **0.41 s / 0.44 s** | 0.00 s | exit 0 |
| 第一次跑（冷启动） | **>5 s 超时** | — | 被 kill |

**第一次超时是假警报**：不是代码死等，是 `docker run` 首次启动 + 镜像层加载超过了 5 秒窗口。同一份代码再跑两次就稳定在 0.4 秒。

这条对平台判定的意义：**如果平台第一次调用觉得慢，那是容器启动开销，不是我们在等 EOF**。骨架里用的是 `for line in sys.stdin`（与官方 `baselines/predict.py:30` 同写法），实测在容器里逐行返回，不会阻塞。

### 改完之后重验两遍

| 位置 | A macro-F1 | B RMSE | C RMSE | 行数 |
|---|---|---|---|---|
| 本地 `harness/predict.py` | 0.7210 | 681.1598 | 50.9320 | 1244 / 1679 / 23049 |
| 容器 `zhixun-harness:1.3 --network none` | 0.7210 | 681.1598 | 50.9320 | 同上 |

数字与改动前逐位相同，说明这次只动了读取循环、没动计算。

**顺带记一个坑**：验证脚本里一开始取 `outputs[0]["label"]` 当预测，得到 macro-F1 **0.0388**（全判 Ethanol）。原因是**分类输出按类别固定顺序排列，不按分数排序**，必须 `max(outputs, key=lambda o: o["score"])`。写评测脚本时要注意。
