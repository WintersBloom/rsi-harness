# SupCon + 度量头移植实验（2026-11-16）

## 问题

外部会话给出的 A 包 v2 方案（逐样本 robust 缩放 + SupCon + 马氏白化原型头）在它自己的实现里
滚动 9 轮均值 0.8381、首轮 0.7475（官方 baseline 0.7140）。问：能不能移植到官方 `train.py`
框架，在**合规前提（从 baseline 出发、不改 checkpoint 格式）**下拿到同样的增益。

## 做法

`out_ext/supcon_lab/` 是官方包的完整副本，只改 `baselines/model.py`：

1. 加入 `_supcon_grad` / `_proj_back` 两个辅助函数（从参考实现复制，逐行未改）；
2. 在 `Model.fit` 的每个 batch 里，把 SupCon 梯度加进反传链路（环境变量 `SUPCON_W`、`SUPCON_TAU` 控制，默认关闭）；
3. 可选余弦原型头（环境变量 `COSHEAD`）。

checkpoint 格式、`validate()`、参数集、训练入口、`predict.py` 全部未动。

## 结果（A 包 batch2 验证集，`--from-baseline checkpoint.npz`）

| 配置 | epochs | macro-F1 | vs baseline |
|---|---|---|---|
| 官方 CE（baseline 公布值） | 20 | **0.7140** | — |
| CE + SupCon（6 维 logits 上算） | 20 | **0.7231** | **+0.91pp** |
| CE + SupCon（6 维 logits 上算） | 60 | 0.7160 | +0.20pp |
| CE + SupCon（32 维 hidden 上算） | 20 | 0.7100 | −0.40pp |
| CE + SupCon（32 维 hidden 上算） | 40 | 0.6973 | −1.67pp |
| robust 缩放（无 SupCon） | 300 | 0.7194 | +0.54pp |

（robust 与官方标准化的完整对照见 `docs/扩展验证.md` 与 `docs/A包_漂移方案v2.md` 复核一节。）

## 结论

**移植能拿到的最好结果是 0.7231（+0.91pp），在 ±1.3pp 的 bootstrap 噪声带内，不构成提升。**

三条原因：

1. **起点不同**。参考实现的 0.7475 是**随机初始化、从头训**；我们必须 `--from-baseline`，
   权重已经被官方的 CE 目标塑形过，SupCon 只是在它旁边加了一个正则项，改动幅度有限。
2. **SupCon 放错层反而更差**。放在 6 维 logits 上（等于直接约束类间角度）比放在 32 维
   hidden 上好 1.3pp——这个任务只有 6 类、线性可分，表征层再做对比学习没有额外信息可挖。
3. **多训就过拟合**。20 epoch 之后所有变体都在掉，官方 `history.jsonl` 也显示同样的形状。

## 不建议改动提交物

`harness/` 与镜像 1.3 已冻结、跑通端到端、数字全部可复现。为 +0.91pp（噪声内）去改
`model.py` 的反传、重跑三条十轮链、重传镜像，风险远大于收益。

## 留档

- 补丁脚本：`tools/exp_supcon_patch.py`、`tools/exp_supcon_fix.py`
- 实验目录：`out_ext/supcon_lab/`（含 `s20/`、`s60/`、`h20/`、`h40/` 四组输出）
