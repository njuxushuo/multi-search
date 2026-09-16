# SFT 优先的下一阶段执行计划

更新日期：2026-09-16

## 当前决策

下一阶段不启动 GRPO/DAPO。现有 Full SFT 已证明 SFT 方向有效，但冷启动数据中 72.05% 的样本只搜索一次，且 Teacher/SFT/评测的 Prompt 和上下文策略并未完全对齐。在这种初始策略上直接惩罚检索次数，很可能加剧过早回答，而不是学到更高效的多跳搜索。

模型起点已确定：27B Teacher 继续使用官方 post-trained `Qwen3.5-27B`；新 Full SFT Student 使用官方 post-trained `Qwen3.5-4B`（即本项目所说的 instruction-capable 模型），不再使用 `Qwen3.5-4B-Base`，也不进行两种初始化的训练对照。主要提升必须相对这个未训练的 post-trained 4B baseline 计算，以保证公平。

近期唯一主线是：

```text
收尾现有基线
    → 实现并核验已冻结的 v3 Search-R1-aligned 合同
    → 修正 scorer 并建立无偏端到端 dev
    → Teacher 恢复采样 pilot
    → 决定局部补采还是全量重采
    → 重建 15k + 1k SFT 数据
    → 从官方 post-trained 4B 训练新 Full SFT
    → 同一新协议下对比
    → 达到准入门槛后再讨论 RL
```

每一阶段只修改明确的变量，上一阶段没有完成验收时不启动后续的大规模训练。

## P0：收尾并冻结现有基线

状态：进行中，不阻塞代码/数据审计。

### 任务

1. 完成正在运行的 Qwen3.5-4B 官方后训练版 v1 50k 评测。注意正式模型名称是 `Qwen3.5-4B`，不得再写成不存在的 `Qwen3.5-4B-Instruct`。
2. 等待其他服务器回传 27B Teacher 完整结果；本机不重复启动 Teacher 评测。
3. 完成后严格校验 50,000 个 eval ID、metadata、重复/缺失 ID 和非空答案评分，再更新 `RESULTS.md`。
4. v0/v1 的原始 JSONL、summary 和日志保持只读，不因后续修改而覆盖。

### 验收

- 每个模型都恰好覆盖固定 manifest 的 50,000 个 ID。
- `RESULTS.md` 只包含完整结果，没有运行中估计。
- 报告同时包含 micro/macro EM、F1、实际 BM25 调用、搜索尝试、格式错误、重复查询和终止原因。

## P1：冻结 canonical 交互合同

状态：设计已冻结、代码待实现；完整规范见 `EXPERIMENT_PROTOCOL.md` 的 v3 章节。这是新 Teacher 采样、SFT 和评测的共同依赖。

### 必须对齐的字段

1. **初始任务说明**：Teacher 生成、SFT 和端到端评测直接读取 parquet 中的 Search-R1 原始 prompt，不再让 SFT 只看裸问题，也不在不同脚本中改写文字。
2. **动作语法**：assistant 每轮只能输出 `<think>...</think><search>...</search>` 或 `<think>...</think><answer>...</answer>`。
3. **连续轨迹**：Qwen chat template 只渲染初始 user prompt；后续 assistant action 和环境 `<information>` 在同一序列中累计，不把 evidence 改成新 user turn。
4. **工具返回**：BM25 top-k=3，使用 `Doc n(Title: ...)` 格式；doc ID/rank/score 和截断前文本写入审计记录。
5. **重复查询**：评测/生成时不立即终止，合法 search 都计 cost；可以用缓存返回相同 top-3。正式 SFT 严格排除含重复或无进展步骤的轨迹，不手术式删除步骤。
6. **搜索预算**：最多 4 次 search action，之后额外给一次 answer-only 机会。
7. **Token 预算**：总上下文/SFT cutoff 8192、每次 assistant 最多 768、每次 top-3 observation 最多 768；所有限制按实际 tokenizer 计算。
8. **错误语义**：重复/无进展与格式错误、空查询、检索异常、生成截断、上下文溢出分开记录。
9. **答案评分**：归一化后的 prediction 和 gold 都必须非空；标准 EM/F1 实现在所有环节复用。

### 上下文策略决策

v3 固定使用 `cumulative_flat_8k`：累计保留问题、所有 action 和实际 evidence。设计包络不超过 7424 token，正常情况下不做历史裁剪。Teacher/SFT 候选若超过 8192 直接以 `sequence_overflow` 拒绝，不能静默截断；评测若出现 overflow 记为失败并报告。`latest_exchange` 只属于历史 v0/v1，不再用于新蒸馏主线。

### 产物与验收

- 新建一份机器可读的 canonical 配置，Teacher/eval/SFT 转换从同一来源读取 Prompt、标签、top-k、轮数和 token budget。
- 对同一题生成 Teacher prompt、SFT conversation 和 eval prompt 的逐字段 diff，除 chat template 的必要包装外不得有语义差异。
- 加入边界测试：恰好 768/8192 token、动作标签位于最后 token、三篇文档均保留、超限显式失败、information labels 全为 `-100`。
- 新协议使用 v3 和独立输出目录；不修改 v0/v1/v2 文件，也不将新结果与 v1 当成同口径数字。

## P2：评分、检索上限和数据隔离审计

状态：在 Teacher pilot 前完成。

### 评分器

1. 在新 online evaluator/reward 中直接修复空字符串 EM/F1 问题。
2. 建立单元测试：空 prediction、只含标点的 gold、`The A`、多别名、日期/数字和正常答案。
3. 分层抽查约 200 条 EM=0，区分真正错答、别名问题、答案抽取问题和数据集官方 evaluator 口径差异。

### BM25 上限

1. 在 HotpotQA/NQ 以及有 supporting evidence 标注的多跳集上统计 top-1/3/5/10 的 answer-string recall 和 supporting-document recall。
2. 按数据集、单跳/多跳、答案类型拆分；不把 BM25 本身无法召回的题归因于 SFT。
3. 记录 query→doc ID 和实际返回文档数，不用“一次 top-k=3”代替真实 doc count。

### 数据隔离

- 为每道题保留稳定 question ID、source、gold answers 和 split。
- 使用 ID/hash 去重，检查 train、teacher-forced eval、interactive dev 和固定 50k final test 无泄漏。
- 固定 50k test 不再用于频繁调参；建立 2k–5k 的无偏 interactive dev，从未用于 SFT 的数据中确定性抽样并保留 gold。

## P3：Teacher 恢复采样 pilot

状态：高优先级；先运行小规模，不直接全量重采。

### Pilot 设计

1. 从 HotpotQA/NQ 按 source、Hotpot type/level 分层取约 2,000 道唯一问题，每题以 4 个 seed 生成候选，第一轮约 8,000 个 rollout。
2. 使用 post-trained 27B Teacher、v3 原始 Prompt、top-k=3、最多 4 次 search、8192/768/768 token 预算；初始采样使用 `temperature=0.7, top_p=0.9`。
3. 对所有候选落盘，不只保存成功轨迹。必须保存原始输出、canonical 序列、逐轮 prompt token 数、query、doc IDs/rank/score、截断前后 evidence、答案、终止原因和 validator 的全部失败标签。
4. 生成环境允许重复/no-progress 后继续，以观察恢复能力；但正式 SFT 严格拒绝含重复、无进展、格式错误、空 query、空结果、生成截断、上下文溢出或最终错答的轨迹。
5. 禁止删除坏步骤后保留其余部分。优先从同一问题的其他 rollout 选择干净轨迹；没有则重新采样。
6. HotpotQA 使用两篇 gold supporting documents 做证据覆盖审计；NQ 统计 gold answer/alias 是否出现在模型可见 evidence 中。

### 必须报告

- 候选数、正确率、格式合规率、可恢复/不可恢复失败率。
- 过滤前后 1/2/3/4 次 search action、实际 BM25、unique query/doc 分布。
- 重复/no-progress 后恢复正确的数量，以及为何仍不进入正式 SFT。
- Hotpot 双 supporting-document 覆盖率、逐轮 support recall；NQ answer-evidence coverage。
- 按 source/type/level 的成功率、轨迹 token p50/p90/p95/p99/max、生成截断和上下文溢出率。

### 采样规模决策门

- v3 Prompt、连续上下文和 observation 格式均与旧 18,453 条不兼容，因此新主线原则上全量重采；旧数据只用于长度、错误分布和对照分析，不能与 v3 轨迹直接混合。
- 先用 pilot 严格合格率估计规模，按 `18,000 / 合格率 × 1.2` 生成候选；若多轮/困难桶不足，再按缺口定向扩采。

不设“每题必须多次搜索”的硬标签，因为一次 top-k=3 可能已完整覆盖两篇支持文档。最终选集软目标为 2+ 至少约 60%、3+ 约 20%–30%；必须同时满足证据新增，不得用冗余 query 填长轨迹。

## P4：重建 SFT 数据

状态：依赖 P1–P3。

### 构建规则

1. 先取得不少于 18,000 条严格合格候选，再按 question ID 去重并排除 interactive dev/final test 泄漏，确定性分层选择 15,000 train + 1,000 teacher-forced eval，至少保留 2,000 条候补。
2. 分层至少包含 source、Hotpot type/level、搜索 action 桶（1/2/3/4）、证据覆盖等级和轨迹长度桶；固定 seed 并生成 manifest/checksum。
3. 样本 metadata 保留 question ID、source、gold answers、Teacher 模型、Prompt/protocol 版本、query/doc IDs、搜索次数、过滤标签和原始轨迹指针。
4. 15,000/1,000 表示 canonical **完整轨迹数**。每条轨迹按运行时顺序编译为单一连续序列，不改写成新的 user/environment 多轮 chat，也不把一条轨迹默认拆成多个独立训练样本。
5. chat template 只渲染 canonical 初始 prompt；环境 `<information>` 直接插入连续轨迹，但其 label 全部为 `-100`。
6. Loss 监督该轨迹内所有由 Teacher 生成的 `<think>`、`<search>` 和 `<answer>`，屏蔽 prompt、全部 `<information>` 和 padding。使用自定义预处理/collator，不能只依赖 `train_on_prompt=false`。
7. Student tokenizer 后序列必须 `<=8192`；任何超限候选以 `sequence_overflow` 拒绝，禁止静默截断目标、证据或标签。
8. 初始选集目标约 HotpotQA/NQ=70%/30%；HotpotQA 优先双 supporting-document 覆盖，保持原始 type/level 分布。一次搜索若同时覆盖双支持文档可以保留。

### 数据 QA

- 100% 结构解析、question ID 唯一性、split 泄漏、最终答案正确和 query→information 对齐检查。
- 按 source/type/level/搜索次数/证据覆盖分层人工抽查，重点检查 2+/3+ 轨迹是否真正获得新证据。
- 在正式训练前对 tokenizer 后 `input_ids/labels` 做自动断言和可读抽样：prompt、evidence、padding 全为 `-100`，所有 Teacher action/answer 为有效 label，没有空监督或越界样本。
- 报告 token 长度 p50/p90/p95/p99/max、截断数、有效监督 token 比例和 1/2/3+ 搜索分布；与旧 15k 表格对比。

### 验收

- 恰好 15,000 train + 1,000 eval canonical 轨迹，全部来自真实 27B Teacher 工具轨迹，并保留至少 2,000 条候补审计轨迹。
- 没有格式错误、答案错误、空/无效工具调用或未记录的静默修改。
- 多轮占比与证据新增统计同时改善，而不是只把平均搜索次数做高。

## P5：从 post-trained 4B 训练新 Full SFT，不重训 LoRA

状态：依赖 P4；只在数据 QA 通过后启动。

### 初始配置

- 从官方 post-trained `/data0/xs/models/Qwen3.5-4B` 独立开始，不在旧 SFT checkpoint 上继续。该模型是已有指令遵循能力的官方后训练 checkpoint，正式名称没有 `-Instruct`。
- Full-parameter SFT + ZeRO-3，GPU 0–3，BF16，gradient checkpointing，`cutoff_len=8192`；若 4 卡 smoke test 显存不足，只调整 micro-batch/ZeRO/offload，不静默把数据截回 4096。
- 8192 只是允许上限；训练动态 padding 到 batch 最长样本并按长度分桶，不把全部样本预填充到 8192。sample packing 只有在边界和 loss-mask 测试通过后才能启用。
- 初始保留有效全局 batch 8、lr 1e-5、cosine、warmup 5%、seed 42，避免在换数据的同时修改过多超参。
- 最多预留 2 epochs；第 1 epoch 后是否继续，由 teacher-forced loss 与 interactive dev 共同决定，不预先强制训满。

### 两层验证

1. **Teacher-forced eval**：固定 1,000 条，每 250 optimizer steps 计算 eval loss，同步 save checkpoint；记录监督 token 数，确保 loss 口径不变。
2. **Interactive dev**：真实调用 BM25 进行多轮 rollout。每 500 steps 在固定的约 1,000 题快速子集上评估；每个 epoch 结束和候选最佳 checkpoint 在完整 2k–5k dev 上评估。

Interactive dev 必须上传 SwanLab：EM、F1、answer rate、format compliance、actual searches、search attempts、progressive/redundant searches、unique docs、repeat rate、search-limit rate、context truncation rate，并按 source 和 1/2/3+ 搜索桶拆分。

### Checkpoint 选择与第二轮决策

- 不再只按最低 eval loss 选模型。主排序为 interactive dev EM/F1；在准确率似然相当时，再用较少的冗余搜索、较高格式合规率和较低预算耗尽率破平。
- 只有第 1 epoch 末 interactive dev 仍在改善、eval loss 未显著恶化且格式/搜索行为无退化时，才从 checkpoint 续训第 2 epoch。
- 若 loss 下降但 EM 不升，先检查输入协议、数据分布和生成行为，不盲目增加 epoch。
- checkpoint 保留策略需要保留至少：最佳 interactive EM、最低 eval loss、每个 epoch 末三类节点，避免 `save_total_limit=2` 提前删掉需要的候选。

## P6：新 SFT 的同协议对比评测

状态：依赖 P5。

1. 主对比在固定 dev 上比较“未训练的官方 post-trained 4B”与“由它启动的新 Full SFT”。Base 和旧 Base-init Full SFT 只作历史参考，不用于计算新 SFT 的主提升；27B Teacher 作为质量上界参考。
2. 只有新 Full 通过 dev 门槛，再对固定 50k final test 做一次完整评测，减少对 final test 的反复调参。
3. 若 canonical Prompt/上下文策略相对 v1 发生改变，所有需要横向比较的模型都在新协议下重测；不用新 Full v3 数字直接减旧 Full v1 数字。
4. 主表同时报告总体与分数据集 EM/F1，全体/正确/错误样本的检索次数，以及失败类型。对 HotpotQA、2WikiMultiHopQA、MuSiQue 单独报告多跳改善。

## RL 准入门槛

以下条件全部满足后，才重新评估 Outcome GRPO → 成本感知 DAPO → 过程优势估计：

1. Teacher/SFT/eval 的 canonical Prompt、角色和上下文策略已冻结，跨环节回归测试通过。
2. 新 Full SFT 在独立 interactive dev 上稳定优于 Base 和旧 Full，尤其在多跳集上有改善。
3. answer rate/格式合规率提高，search-limit/重复循环率降低；正确样本中有足够的 2+/3+ 轨迹，存在可优化的成功路径长度差异。
4. 非空 EM/F1、格式奖励、搜索成本和状态等价逻辑均有单元测试，不再使用旧 Search-R1 的空串/答案抽取脆弱实现。
5. 已完成新版上游 veRL + Qwen3.5/vLLM 的小规模 rollout/weight-sync/checkpoint 恢复 smoke test。

满足后也不一次同时引入所有奖励：先做 outcome-only 小试验，再加组内最短成功路径成本奖励，最后单独评估过程级 advantage。

## 当前明确不做

- 不立即启动 GRPO/DAPO/RL，不在数据分布未修复时先压低搜索次数。
- 不重训 LoRA；它已完成诊断使命，现有 v1 结果支持把资源用在 Full SFT。
- v3 协议冻结并通过 pilot 后，不在同一次正式训练中继续修改 Prompt、top-k、token budget、学习率、batch size 和 epoch；若必须改变则建立独立实验，避免无法归因。
- 不为达到简历中的 2.59 次而人工填充搜索；只接受能提供新证据或完成真实多跳的较长轨迹。
- 不用 1k Teacher 成功轨迹的 loss 替代端到端 EM，也不只根据最低 loss 选 checkpoint。
- 不频繁查看固定 50k final test 调参；日常决策使用独立 interactive dev。

## 每个新实验的最小记录

- 唯一实验 ID、Git commit、dirty diff、日期、随机种子。
- 模型/数据/manifest/Prompt/protocol/tokenizer/chat template 的路径和 checksum。
- GPU、Python/CUDA/PyTorch/Transformers/vLLM/LLaMA-Factory 版本。
- 完整启动命令、训练配置、SwanLab run ID、日志和 checkpoint 路径。
- 数据 QA、loss-mask QA、端到端指标及失败分布。
- 恢复训练时记录恢复 checkpoint 和全局 step，不将重启后的局部 loss 序列当成新实验。
