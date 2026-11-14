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
