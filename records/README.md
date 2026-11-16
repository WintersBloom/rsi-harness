# 训练记录（父子链证明）

这个目录放三条十轮链每一轮的 `training.json`（官方 `baselines/train.py` 自动生成），共 **24 份**：A 链 8 轮、B 链 6 轮、C 链 10 轮。

## 为什么放在这里

平台公告明确写了「**禁止从头训练冒充续训**」，而 `TRAINING_CHECKPOINT` 的 `parent_checkpoint_sha256` 是唯一能自证的字段。提交的 `.npz` 里**不带**这个字段（checkpoint 的 metadata 只有 `epochs / learning_rate / steps / paths` 等），所以必须把 `training.json` 一起留着，否则无法证明每一轮都真的从上一轮出发。

## 校验结果

用脚本逐轮检查每个 `parent_checkpoint_sha256` 是否落在「官方 checkpoint 或已出现过的某一轮」这个集合里：

| 链 | 轮数 | 全部 parent 可追溯 |
|---|---|---|
| A（气体分类） | 8 | ✅ |
| B（轴承 RUL） | 6 | ✅ |
| C（发动机 RUL） | 10 | ✅ |

A 链的起点对得上官方的 `checkpoint_sha256`：

```
官方 A checkpoint:  sha256:fd1e08b2aad08b2e9754f9b2dfdc029481c0da7ecdedca0abb0ec6fcc818c757
   ↓
chain_A_r01  parent = 上面这个        self = sha256:8aa73140cada...
   ↓
chain_A_r02  parent = 8aa73140...     self = sha256:f972ef52...     ← 被评估门回退
chain_A_r03  parent = 8aa73140...     self = sha256:e3c9bf73...     ← 从 r01 重新出发
```

**r03 的父不是 r02，是 r01**——因为 r02 被评估门判定为退步、回退了。这正是"评估门"该有的行为：被拒绝的那一轮不进入父链。所以链条是**一棵树，不是一条直线**；校验脚本判的是"每个父都在已知集合里"，不是"父恰好是上一轮"。

## 复现

```bash
# 每一轮都是这样产生的
python -m baselines.train --profile A \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 20 --output out/r01
# 之后每轮用上一被接受轮的 checkpoint.npz 作为 --from-baseline
```

`--from-baseline` 与 `--resume` 在官方 `train.py` 里是同一套续训语义（`baselines/train.py:74` 强制 `resume must preserve the checkpoint's random seed`），所以父子 sha 一定接得上。
