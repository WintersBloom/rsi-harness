# 智训引擎 · 工业制造场景下的模型训练与超参数优化

**提交物**：一个能在断网环境下逐行吃数据、逐行吐答案的评测镜像；一套在官方三包上各跑十轮、每轮留证的训练链；一份如实的实验记录。

**依赖**：Python 3.10+，**零第三方库**（只要 numpy）。不需要 GPU，不需要联网。

---

## 当前交付状态（2026-11-14）

| 项 | 状态 | 说明 |
|---|---|---|
| ① 评测镜像 | **完成** | `zhixun-harness:1.3`（345 MB，限 2 GB）；提交物是 `dist/zhixun-harness-1.3.tar`（86.3 MB，sha256 `A494FEC45F6223F8DF9917011F97DD27628A83AD50503D25A63B18F34855375F`）。**这个 tar 已锁定为提交物**：`docker save` 每次都会带新的时间戳、哈希必变，所以不要再覆盖它——代码若再改，重打并同步更新这一行 |
| ② SwanLab 记录 | **已设公开（用户确认）** | 12 个 run：[zhipin-engine](https://swanlab.cn/@WintersBloom/zhipin-engine)（9 个合成流横评 + 3 个真实数据十轮链）。自动化探测（HTTP 200、无登录墙）无法判定可见性，**提交前请用无痕窗口打开一次确认** |
| ③ 运行说明 | **完成** | 本文 + `docs/提交说明.md`（评审入口） |

### 三个任务的成绩

| 任务 | 官方基线 | 我们提交的 | 变化 |
|---|---|---|---|
| A 六类气体（Macro-F1 ↑） | 0.7140 | **0.7210** | +0.70pp（+0.98%），**在噪声带内**（bootstrap SE ±1.3pp，差值 0.55 个标准误，统计上不显著） |
| B 轴承 RUL（RMSE ↓） | 681.16 | **681.16** | 不动（六个动作全使 RMSE 变大，主动保留官方权重；根因见 `docs/作品文档.md` 第三节） |
| C 发动机 RUL（RMSE ↓） | 52.87 | **50.93** | −3.66% |

分数由容器实测复现，不是本地算的：`docker run --network none -i zhixun-harness:1.3 < inputs.jsonl`。

### 三十秒跑通

```powershell
docker load -i dist/zhixun-harness-1.3.tar
docker run --rm --network none -i zhixun-harness:1.3 < inputs.jsonl > outputs.jsonl
```

输入一行一个 JSON：`{"inputs": {...}}`（不需要 `task` 字段，路由靠 `inputs` 的形状，已用三包全量验证零误判）。
输出一行一个 JSON：分类 `{"outputs":[{"label","score"}...]}`、回归 `{"outputs":{"value":x}}`。

### 这个系统在做什么

平台不给即时标签（`baselines/predict.py:1` 写着 "no model execution hooks"，测试里只要输入带 RUL 字段就直接抛错）。所以"自进化"不在运行时发生，而在**训练轮**发生：每一轮先诊断当前权重错在哪，再决定改什么，训完在验证集上判一次，变好才采纳、变差就回退。

十轮里 A 有 6 次回退、B 有 6 次全回退、C 有 4 次回退。**"动了不如不动"不是失败，是这套判据在起作用**——B 包六个动作全是坏的，系统一次都没采纳，最终保持官方权重，这是正确结果。

### 诚实边界

- **A 包的 Acetaldehyde 恒为 0.0000**。不是没调，是换批后它与 Ethanol 的相对位置**翻了向**（跨批次二分类 AUC 0.1137，取负才 0.886；同批次内部 5 折 AUC 0.9939）。在"验证集不得参与拟合任何统计量"的红线下，方向翻转无法恢复。五组重平衡实验全部确认（见 `docs/a_acetaldehyde.md`）。
- **B 包没有增益**，如实报 0。
- 评分表的"归一化增益 G"算不出分母（官方没定义"阶段"），我们不自造口径，只报绝对增益与单位成本。

---

## 以下为原型阶段记录（数据窗闭环，2026-11-06 至 11-11）

数据包到手前，我们用自己生成的漂移流把整套闭环跑通了。这部分**不作为提交物**，保留是因为它解释了"为什么判据是这个形态"。若要读，从 `docs/experiments.md` 开始。

**原型的核心结论**：五环闭环跑通、十组对照完成（含"周期重训""总是重训"两个诚实对照）、幂等已验证；等官方数据包替换合成数据。
**依赖**：Python 3.10+，**零第三方库**（只要 numpy）。不需要 GPU，不需要联网。

## 原型提交状态（历史，已被上面替代）

三件套里还剩一件没到位。**不粉饰**，逐条写清楚：

| 项 | 状态 | 具体缺什么 | 怎么补 |
|---|---|---|---|
| ① 模型配置 + Harness 镜像 | **完成** | `dist/zhipin-0.7-offline.tar`（86.1 MB，sha256 `4376B2F9…`）**从零构建**：`docker build --network none --no-cache -f Dockerfile .` **8.0 秒**跑通（依赖来自仓库自带 wheel，构建期零联网），`docker run --network none` 离线运行，F1 0.8208 / AUC 0.9822 / AP 0.9394。另在 **Ubuntu 24.04 + python3.12 + apt 版 numpy** 上跑出同一组数字（跨发行版验证，见 docs/image_verify.md I 节） | — |
| ② SwanLab 云端实验记录 | **已设公开（用户确认，建议无痕窗口复核）** | 9 个 run 已同步到 **[swanlab.cn/@WintersBloom/zhipin-engine](https://swanlab.cn/@WintersBloom/zhipin-engine)**（每 run 237 条记录，含 21 窗逐点曲线）。**还差一步：该项目当前是私有的，需要在网页上手动设为 Public**，否则评审打不开链接（`swanlab api project` 只有 create/info/list，改不了可见性） | 登录后进项目 → 设置 → 可见性改为公开 |
| ③ 系统运行说明文档 | **基本完成** | 本文 + `docs/` 下 6 篇 | — |

**这三条里 ①③ 已完成，② 只差"点一下公开"。** 本地留痕不算留痕——上传完成但链接打不开，评审看到的效果和没上传一样。

**另外一件没做到的**：评分表"漂移流综合性能 15 分"用的归一化增益 G，我们**算不出来**——自己造的漂移流里，"全量重训"这个天花板反而比冻结基线还差（分母为负）。机制、公式、工具（`tools/build_g_stream.py --data`）都已就绪，官方数据一到 1 小时内出数。详见 `docs/g_stream.md`。

**还有一件必须自己承认的**：不看漂移、每 4 窗无脑重训一次（I 组）F1 0.8252，比我们的 0.8229 **还高 0.23pp**。我们的触发机制在这条流上没赢过它，只赢了算力（0.81s vs 1.44s，省 44%）。见 `docs/experiments.md` 第 2.1 节。

## 分批下发怎么跑（11/06 培训新增的约束）

官方口径：「**必须先输出当前批次"用哪个模型 / 不上线"的判断，才能拿到下一批数据**」，
且「每次下发会把**之前所有数据连同标签再给一次**」。

第二条是关键：拿到的是**累积数据**，不是增量。所以本系统的适配方式是：

```bash
# 第 k 批数据到手（含前 k-1 批的全部行 + 标签），直接跑：
python src/main.py --data batch_k.csv --outdir results/batch_k
# 输出：results/batch_k/run_summary.json 里最后一个窗口的 decision 就是"本批用哪个模型/不上线"
```

**为什么不用改架构**：每个窗口的决策只依赖"当前窗口数据 + 之前窗口的模型状态"，
重跑历史窗口得到的状态与当初完全一致（幂等已验证，见第六节、`docs/image_verify.md` E 项）。
所以"每次都从头跑累积数据"和"增量续跑"在结果上等价，只是多花时间。

**时间够不够**（4 核 8G 实测折算）：

| 累积行数 | 窗口数 | 预计耗时 |
|---|---|---|
| 12 000（3 批） | 21 | ~1 s |
| 60 000（10 批） | 105 | ~5 s |
| 300 000（50 批） | 525 | ~25 s |

即使 50 批也远低于任何合理时限。**但如果官方单批时限低于 60 秒且批次数很多，
就需要真增量续跑**——目前 `--resume` 只保留文件、不恢复窗口进度（见第六节），这是已知缺口。

**还有一个必须核对的地方**：官方说"先输出判断才能拿下一批"。如果这个"输出"是通过
评测平台的 API/SDK 提交（而不是我们跑完打印），我们需要知道接口格式才能对接。
`src/main.py` 的决策逻辑已经按"每窗一个 decision"组织（`window_records[idx].decision`
+ `action`），加一层薄 wrapper 就能对接任何 JSON 契约。**这条要等 11/09 平台上线后确认。**

> **用哪个 python**：本机 PATH 里的 `python` 是 msys2 版本，会报 `No module named 'types'`，这是开发机特有的坑。开发时显式指定解释器（本机可用 DSH 自带 runtime 的 `python.exe` 或 `C:\Python311\python.exe`）。**镜像内不受影响**——`python:3.11-slim` 里的 python 正常，评测方按 `docker run` 跑不会碰到这个问题。

---

## 一、30 秒跑通

```bash
# 1) 生成合成漂移流（官方数据包到位后跳过这步，直接把 csv 放 data/）
python src/simulate_drift.py --out data/synth_stream.csv --seed 7

# 2) 训练静态基线（= 主办方那个"部署后不再更新"的模型）
python src/baseline.py --data data/synth_stream.csv --out experiments/baseline_static.json

# 3) 跑完整闭环（监测 → 归因 → 决策 → 执行 → 评估 → 回滚）
python src/main.py --data data/synth_stream.csv
```

第 3 步会在 `results/` 下产出：

| 文件 | 内容 |
|---|---|
| `run_summary.json` | 总体指标 + 三口径（均值/末窗/最差窗）+ 逐窗口明细 |
| `run.log` | 每个窗口的决策与理由（可直接贴进报告） |
| `work/ledger.jsonl` | 成本账本：每次动作的 CPU 秒数、前后指标、增益 |
| `work/predictions.csv` | 逐样本预测（可复算任何指标） |
| `work/registry/` | 模型版本快照，支持回滚 |
| `work/checkpoint.json` | 只写不读：记录当前窗口进度，**不用于续跑**（见第六节） |

## 二、对照实验

```bash
python tools/run_suite.py --seeds 0,1,2 --outroot results/suite3   # 八组策略横评 → results/suite3/compare.md
python tools/drift_sensitivity.py                                  # 温和/基准/剧烈三档漂移下的稳健性
python tools/ledger_report.py --compare                            # 成本账本：每花一秒 CPU 买到多少点
```

> `results/suite` 与 `results/suite2` 是**历史产物，已作废**。第二轮验收发现两处时序问题（评估门读到了当窗、账本把告警等级当 F1 累加），修完之后决策全变了，旧表不能与新表相减。

结果与结论见 [docs/experiments.md](docs/experiments.md)。

## 三、参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | `data/synth_stream.csv` | 输入流，列名 `t,segment,f*,label` |
| `--window` | 500 | 滑动窗口大小 |
| `--train-n` | 1500 | 初始训练样本数 |
| `--train-window` | 3 | 重训时用最近几个窗口（滑动窗口） |
| `--budget-cpu-s` | 60 | 本次运行算力预算，用完强制停手 |
| `--tolerance` | 0.005 | 评估门容差，新模型退化超过它就被拒 |
| `--guard-margin` | 0.02 | 基线守护：基线领先多少才回滚 |
| `--guard-mode` | `edge` | 守护信号：`edge` 警报解除即回滚（默认，零标签零延迟）/ `f1` 延迟标签 / `dist` 预测分布 / `recover` 分布恢复 / `off` |
| `--weight-mode` | `none` | `recency` = 越新的窗口权重越大（实测无益，默认关） |
| `--frozen` | off | 冻结模式，只监测不更新（跑静态基线对照用） |
| `--resume` | off | 保留 workdir 残留（默认清空，保证重跑结果一致） |
| `--policy` | `rule` | 决策者：`rule` 规则表 / `llm` 每窗问 Qwen / `hybrid` 规则打底、只在边界问 / `auto` |
| `--llm-base-url` | 空 | OpenAI 兼容端点（如 `http://127.0.0.1:8000/v1`）；**留空 = 强制 rule，绝不联网** |
| `--llm-model` | `qwen3.8-27b` | 模型名 |
| `--llm-api-key-env` | 空 | 从哪个环境变量读 key（不落盘、不写进 summary） |
| `--llm-timeout` | 20.0 | 单次调用超时（秒），超时按失败计，触发降级 |
| `--llm-think` | off | 允许模型先思考（默认关：Qwen3 开思考约 450 token/次且可能只回 `reasoning`，关掉 6 token、0.2s） |

## 四、五环闭环怎么实现的

| 环节 | 文件 | 做法 |
|---|---|---|
| 漂移监测 | `src/monitor.py` | 特征 PSI + 预测分布 PSI，双证据门（两侧都异常才升级） |
| 诊断归因 | `src/attribute.py` | 用"特征 PSI vs 预测 PSI"的组合判概念漂移/协变量漂移/混合 |
| 决策 | `src/policy.py` | 可替换的决策层：`RulePolicy`（默认，确定性规则表）/ `LLMPolicy`（每窗问 Qwen）/ `HybridPolicy`（规则打底，边界才问 LLM）。三者共用同一套安全闸与降级路径 |
| 执行 | `src/actions.py` | 阈值校准（无监督）或滑动窗口重训 |
| 评估上线 | `src/main.py` | 用"标签已到达的上一窗"评估门，不达标就拒（等效回滚） |
| 回滚 | `src/registry.py` | 版本快照 + 基线守护哨兵 |

**时序约定（这是全系统的关键前提）**：一个窗口的预测**必须**只用该窗口之前的数据；窗口收集完毕、标签（人工复判）到达之后，才允许重训。这条线一旦破了，所有增益都是假的——第一版就是在这里出的错（详见 `docs/experiments.md` 的"泄漏消融"一节）。

**LLM 在哪、为什么默认不用**：赛题限定"驱动 Harness 的大模型"只能是 PyTRiO 自训模型或 Qwen3.8-27B（`out2.txt:57-58`），但这个模型跑在 **PyTRiO 侧**，我们的容器只是客户端——所以"8G 装不下 27B"不是理由。真正的约束是 `out2.txt:106`：**PyTRiO 训推额度只在企业赛阶段提供**，校选赛阶段调不到，也无法在本机验证。

因此默认 `--policy rule`（确定性规则表，0 Token、0.8 s/窗、可审计），但**决策层是可替换的**：`--policy llm` 每窗问一次 Qwen，`--policy hybrid` 规则打底、只在"规则拿不准"的边界问。三种实现共用同一套安全闸，并带三级降级：

1. 没给 `--llm-base-url` → 直接退回规则；
2. 单次调用失败 → 本窗退回规则（`source=rule_fallback`）；
3. 连续失败 2 次 → **熔断**，剩余窗口全走规则，不再拖慢。

实测：不给 URL / 给死端点两种情况下，F1 都是 0.8208，与纯规则逐窗动作完全一致——**"没有 LLM 也能跑完"这件事是被测过的，不是承诺**。

这样"用不用大模型"就变成了一个可做对照实验的开关（规则 / 纯 LLM / 混合三方），而不是架构上的赌博——正好对应主观分"技术路线选择与论证 6 分"。

## 五、离线与可复现

评测方拿到的是 tar，不需要构建，也不需要联网：

```bash
docker load -i dist/zhipin-0.7-offline.tar
docker run --rm --network none -v "${PWD}/out:/out" -v "${PWD}/data:/data" zhipin-0.7-offline --data /data/stream.csv
```

（`${PWD}` 是 PowerShell 写法；bash 下用 `$PWD`。镜像内的 `--outdir /out --workdir /out/work` 已写进 ENTRYPOINT，只需要给 `--data`。）

自己构建（需要联网拉基础镜像，或用本地已有的 `python:3.11-slim`）：

```bash
docker build -t zhipin-0.7-offline .
```

- 无网络：不下载任何东西，不调用任何 API。
- 无 stdin：全程无人工输入。
- **确定性**：同 `--seed` 同数据，两次运行的指标与 `predictions.csv` 完全一致（已验证）。
- **幂等**：每次启动自动清空 `registry/`、`ledger.jsonl`、`checkpoint.json`，重跑不会叠加旧状态。

## 六、目录与现状

```
src/            全部源码（纯 numpy，零依赖）
tools/          实验脚本（八组横评、漂移敏感性、监测器探针、账本报告、G 估计器、SwanLab 回放）
docs/           实验报告、监测器标定、G 流尝试、SwanLab 上传说明、答疑清单
experiments/    基线参照（baseline_ref.json）
data/           数据（当前是合成数据）
results/        运行产物（suite3 是当前唯一有效的一批）
dist/           交付物：镜像 tar、SwanLab 离线记录 zip
.team/          四人协作（规划/执行/验收/讲解）的协议与记录
```

**交付物三件套**

| # | 要求 | 位置 | 状态 |
|---|---|---|---|
| ① | 模型配置 + Harness 系统镜像 | `dist/zhipin-0.7-offline.tar`（86.1 MB，从零构建，`docker load` 后离线可跑） | ✅ |
| ② | SwanLab 云端实验记录 | [swanlab.cn/@WintersBloom/zhipin-engine](https://swanlab.cn/@WintersBloom/zhipin-engine)（9 run × 237 records） | 🟡 **已设公开（用户确认，建议无痕窗口复核）** |
| ③ | 系统运行说明文档 | 本文件 + `docs/` 全部 | ✅ |

**SwanLab 云端链接**：`https://swanlab.cn/@WintersBloom/zhipin-engine`（需用无痕窗口确认可打开——自动化探测判定不了可见性）。

**还没做完**：

1. **官方数据包没到位**（平台 11 月初上线）——现在的结论全部基于合成数据与三个公开数据集，只能证明"机制跑得通"，不能证明"在赛题数据上能赢"。
2. **断点续训未实现**：`checkpoint.json` 只记录不恢复，崩溃后是从零重跑（靠幂等保证结果一致），不是从断点继续。别在文档里写成"续跑"。
3. **SwanLab 只差设为公开**：9 个 run 已同步（每 run 237 条记录、含 21 窗逐点曲线），但项目当前私有。`swanlab api project` 只有 create/info/list，改不了可见性，必须网页操作。链接打不开 = 门槛②仍为零。
4. ~~镜像不能从零重建~~ **已解决**：`docker.m.daocloud.io` 镜像站可拉到 `python:3.11-slim`，`docker build -f Dockerfile .` 完整跑通，产出 `zhipin:0.5`；删掉本地镜像后 `docker load` 回来重跑，指标逐位一致（证明 tar 自包含，评测方无需联网）。构建步骤见 [docs/image_verify.md](docs/image_verify.md) F 项。**更强证据（23:5x）**：Dockerfile 已改为**离线 wheel 构建**，`docker build --network none --no-cache -t zhipin:0.7-offline .` 断网重建 **8.0 秒**成功，断网运行 F1 0.8208；产物 `dist/zhipin-0.7-offline.tar`（86.1 MB，sha256 `4376B2F9…`），这是**最终交付的那个 tar**。另用 `ubuntu:24.04 + apt python3-numpy` 跑出同一组数字（跨发行版，见同文档 I 节）。
5. **数据喂入方式未确认**：现假设"挂载 CSV + `--data` 参数"。若评测方用流式 stdin 或 SDK，入口要重写。
6. **归一化增益 G 算不出来**：官方 15 分要 G = (ours−baseline)/(ceiling−baseline)，我们自造的四条流里有三条分母为负、一条 ours 与 ceiling 相等。原因与证据见 [docs/g_stream.md](docs/g_stream.md)。工具已就绪，等官方流。
