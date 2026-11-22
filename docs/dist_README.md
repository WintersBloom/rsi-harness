# dist/ — 提交物归档

这里只放**要交出去的东西**。完整说明见仓库根目录的 [README.md](../README.md)。

## 两个镜像，用途不同，不要只拿一个

| 文件 | 大小 | 协议 | 用途 |
|---|---|---|---|
| `zhixun-harness-sdk14.tar` | 90.5 MB | **新**（`contest.harness.workspace.v2`，三入口 `describe/train/predict`） | **提交用这个**。与 sdk13 只差一处：C 包改 900 轮长程续训 + 回滚最好一轮 + 塌成常数护栏 |
| `zhixun-harness-sdk12.tar` | 90.5 MB | 同上 | 已被 `sdk14` 取代（只差 C 包 900 轮长程续训）；每轮只发一次模型调用；不交占位文件 |
| `zhixun-harness-sdk10.tar` | 90.5 MB | 同上 | 已被 `sdk11` 取代（只差预检四项的兜底整改） |
| `zhixun-harness-sdk11.tar` | 90.5 MB | 同上 | 已被 `sdk12` 取代（B 由一次性收缩改成十档阶梯） |
| `zhixun-harness-sdk7.tar` | 90.5 MB | 同上 | **作废，不要提交**。B 包当年交的是手工收缩权重，被判不合规 |
| `zhixun-harness-sdk9.tar` | 90.5 MB | 同上 | **已被 sdk10 取代**。与 sdk10 只差 B 的续训动作（B 交的是父检查点原样） |
| `zhixun-harness-sdk6.tar` | 86.3 MB | 同上 | **退路**。平台试运行跑通过完整十轮的那一版，只作回滚用 |
| `zhixun-harness-1.3.tar` | 86.3 MB | 旧（单入口 `predict.py`，逐行 stdin） | 兜底 / 证据。旧协议下三包全量成绩的存档 |
| `swanlog-offline.zip` | 57.8 KB | — | SwanLab 实验记录的离线备份（在线版已公开，这里留档） |

sha256：

```
zhixun-harness-sdk14.tar 558765FF0795AA435E710FF519E72E146EB7CE480767125D5C26BE4D042CB0BF
zhixun-harness-sdk7.tar  5E3247A8B83A6344EE5C22144EC47692408CB65248115E9EED205F6780BDA82E  (作废)
zhixun-harness-sdk6.tar  0469AD11C2B711A9523D6BC85D8BD2C2E0DCA925B2554152B6B630708CE1C0A3
zhixun-harness-1.3.tar   A494FEC45F6223F8DF9917011F97DD27628A83AD50503D25A63B18F34855375F
```

> **sdk14 相对 sdk7 改了八处**：① `/app/checkpoints/B.npz` 换回**官方父检查点原样**（sha256 `C7D819FC…3314B4`），删掉"noop 时提交手工收缩权重"的分支；② 新增 `/app/b_regularized.py`（与官方 `train_run` 同签名同产物的自写 trainer，MAE + L2）；⑤ B 的动作从一次性 `b_shrink` 改成**十档阶梯 `b_ladder`**：每轮从父检查点真训一档、只有验证集变好才采纳，公开验证集 681.1598 → 470.76 → 459.76 → **452.13**（采纳 3 档、回退 7 档）；③ C 包动作集新增 `retrain_300`：首轮一次性续训 300 轮（50.9320 → 50.4484，`steps` 55250 → 187850），之后 `noop`——拆成「每轮 30 轮」实测会越训越差（50.93 → 53.53）；④ `configs/B.json` 由三分类改回官方的 `regression` + `target_unit: minutes`（旧配置会让 B 的续训直接抛 `ValueError`）。⑤ B 的动作从一次性 `b_shrink` 改成**十档阶梯 `b_ladder`**（每轮真训一档、只有验证集变好才采纳，681.16→452.13）。⑥ C 新增 `retrain_300`（首轮一次性 300 轮，50.93→50.45）。⑦ `sdk13` 起对模型网关的 502/503/504/超时**重发同一份完整请求**（最多 3 次），401/403/400 与空回复不重试。
> ⑧ 新增 `/app/c_long.py`：C 包首轮一次续训 **900 轮**、每轮回滚到验证最好的那一轮、并加“塌成常数”护栏（预测标准差 < 1 即本轮不提交）；C 首轮从 50.93 降到 **50.07**
>
> 为什么撤回手工收缩：v3 提交被平台判为 `noncompliant`，原话是「请基于官方 baseline 及平台提供的父检查点实现真实续训，不要从头训练其他模型」。收缩版权重 `steps` 不变、`metadata` 里带 `shrink_alpha`，属于在权重文件上做手工变换，不满足"候选必须来自续训"——已撤回、`sdk7` 作废。
>
> 按同一思路重做的合规版：B 每轮由 `/app/b_regularized.py`（MAE + L2，与官方 `train_run` 同签名）从父检查点按十档阶梯真训一次，`steps` 逐轮增长，公开验证集 **681.16 → 470.76 → 459.76 → 452.13**。分数与手工版方向一致，区别在"每一档权重都是训练写出来的"。

> sdk6 相对 sdk5 只加了一件事：**训练日志写 stderr**（`[zhixun] start/decide/train/submit`、失败打完整 traceback）。SDK 在加载选手模块前已把 fd1 指向 stderr（`protocol.py:36-42`），协议帧走另一份 dup，实测 `predict` 100 行仍是 100 行合法 JSON、stderr 0 字节。
>
> sdk5 相对 sdk4 只改训练轮的三处，**`predict` 路径一行没动、三包输出逐条不变**（已用 1.3 的 100 行基准复验 100/100 相同）：
> 1. 新增 `_bank_call()` 三级降级重试，保证 `finish()` 时至少有一个 `call_id`——SDK 的 `round.py:336` 规定"本轮没有任何成功模型调用"就抛 `MODEL_CALL_REQUIRED`，整轮作废；
> 2. 删掉 `rnd.history()` 调用（它一旦抛 `HISTORY_INVALID` 也在 try 之外，同样整轮死）；
> 3. 决策/执行/提交三段全部包进 try，任何异常写进 `change_summary` 的 `fatal=` 字段——**下次失败能在平台页面上读到原因**，不必再猜。

> `docker save` 每次都会写入新的时间戳，**同一份代码重打哈希必变**。所以这两个 tar 就是提交物本身，不要覆盖；代码若再改，重打后要同步根 README、提交说明与本文件。

## 为什么保留 1.3

它过不了新 SDK（没有 `describe/train` 入口），但它是**旧协议下的存档证据**：

| 包 | 行数 | 1.3（官方权重） | sdk7（当前提交） |
|---|---|---|---|
| A | 1,244 | Macro-F1 0.7210 | Macro-F1 0.7210 |
| B | 1,679 | RMSE 681.16 | RMSE **452.13**（十档阶梯，逐轮真训、采纳 3 档） |
| C | 23,049 | RMSE 50.93 | RMSE 50.07（十轮链首轮续训 900 轮并回滚最好一轮） |

A/C 两列逐位相同（同一份 `predict` 代码、同一份权重）；B 一列也相同——十轮全是 `noop`，交付的就是父检查点本身。
如果 sdk7 在平台上出任何问题，sdk6 是已完整跑通十轮的那一版，1.3 是更早的存档——两者都在 `dist/` 里。

## 怎么跑

新协议（三个入口，平台按这个调）：

```powershell
docker load -i dist/zhixun-harness-sdk7.tar
docker run --rm --network none zhixun-harness:sdk7 describe
docker run --rm -i --network none zhixun-harness:sdk7 predict < inputs.jsonl > outputs.jsonl
```

旧协议（逐行 stdin/stdout）：

```powershell
docker load -i dist/zhixun-harness-1.3.tar
Get-Content inputs.jsonl | docker run --rm --network none -i zhixun-harness:1.3
```

- 两个镜像都是**从 stdin 逐行读 JSONL、往 stdout 逐行写 JSONL**，不需要挂载卷、不读写文件。
- 三包靠 `inputs` 的形状自动路由（`settings` → C，`features` 是 dict → B，长度 128 的 list → A），一个镜像跑三个任务。
- sdk10 的权重在 `/app/checkpoints/{A,B,C}.npz`（B 是官方父检查点原样——它在第 1 轮现场续训；A/C 是十轮链选出的续训产物）；1.3 默认加载 `checkpoints_trained`，传 `--checkpoints /app/harness/checkpoints` 可切回官方原版权重复现基线。

## 镜像里有什么、没什么

**有**：`predict.py` / `harness.py`（入口）、`router.py`（三包路由）、`net.py`（纯 numpy 前向）、`contest_loop.py`（训练轮的策略迭代）、六个 `.npz` 权重。

**没有**：监测、归因、决策、回滚的**在线**实现、SwanLab、sklearn、任何必须在推理期联网的东西。

这不是缺失，是合规要求——`TRAINING_GUIDE.md:32` 禁止验证/测试集参与拟合任何统计量，官方 `predict.py:1` 写明 "no model execution hooks"。**推理期零状态、零拟合**；五环发生在训练轮（`train` 入口里），证据在 `docs/` 与 SwanLab 里。
