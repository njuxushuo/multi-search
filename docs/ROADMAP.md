# 多轮检索问答复现实验方案

## 目标与验收口径

复现 Qwen3.5-27B Teacher → Qwen3.5-4B Student 的多轮 BM25 检索问答链路。正式结果必须包含 15,000 条真实搜索轨迹、独立 1,000 条 SFT eval，以及固定 50k 跨数据集测试子集。核心指标为 EM、F1、平均检索次数和无效工具调用率；所有实验保存配置、代码版本、随机种子、日志和 checkpoint。

## 近期实验计划（2026-09-15，SFT 优先）

本节是近期执行顺序；下方阶段 6–9 的 RL 工作暂不启动。当前首先把评测口径和 SFT 数据做好，避免同时修改多个变量。

### P0：冻结 v0 四模型基线

1. 按 `docs/EXPERIMENT_PROTOCOL.md` 中的 v0 协议保存 Base、LoRA、Full SFT、27B Teacher 的逐条输出、summary、日志和实际配置。
2. 当前 Base、LoRA 已完成；继续完成 Full SFT 和自动排队的 Teacher。
3. v0 文件只读保留，不因后续修复而覆盖。
4. 正式结果同时报告各数据集、50k micro average，并补算七数据集等权 macro average；搜索指标至少区分成功搜索、搜索尝试、返回文档数和去重文档数。

验收：四模型恰好覆盖同一固定 manifest 的 50,000 个 eval ID，所有 shard 完整合并，协议 metadata 和文件路径可追溯。

### P1：修正重复搜索提前终止并做局部续测

1. 保持 v0 的模型、manifest、Prompt、BM25、top-k=3、最大 8 轮和解码参数不变。
2. 归一化查询完全重复时不再立即终止：不重复请求 BM25，返回固定纠错提示，引导模型改写查询，并消耗一次搜索尝试预算。
3. 搜索尝试预算固定为 8；第 8 次尝试处理完后额外保留一次只用于最终 `<answer>` 的生成，因此最多 9 次 assistant generation。
4. 不同查询返回相同文档时继续保留 v0 的 no-progress 标记，但不终止；总搜索尝试预算防止循环。
5. 仅处理 v0 中 `repeated_search` 和“8 次搜索后无回答机会”的 `max_rounds` ID。优先从已保存的停止状态续接；若必须从头生成，则固定确定性解码并单独标记，避免把随机漂移归因于协议修复。
6. 普通格式错误与合法但错误的最终答案仍然停止，不提供 gold 正误反馈。
7. 未受影响样本原样复用，合并为单独的 v1 corrected 结果，不能覆盖 v0。

必须报告：受影响样本比例、续测答对率、全量 EM/F1 增量、额外搜索次数、各终止类别的恢复贡献。

### P1：加入 Qwen3.5-4B-Instruct 基线

1. 使用修正后的 v1 协议，不再为 Instruct 单独运行有已知缺陷的 v0。
2. 使用模型官方 chat template，但任务文字、工具标签、检索器、top-k、轮数、stop strings、解码参数和评分器与 v1 其他模型一致。
3. 先在每个数据集固定 500–1,000 条上试跑，检查格式、速度和搜索行为；正常后再运行完整 50k。
4. 重点比较 EM/F1、全部样本与正确样本搜索次数、零搜索率、重复率、格式错误率及最大轮数终止率。

验收：可以回答 Instruct 是否比 Base 更主动搜索，以及增加的搜索是否带来准确率或证据覆盖收益，而不只比较一个平均搜索次数。

### P1：两个低成本审计

1. **BM25 检索上限**：统计 gold answer 或有标注的 supporting evidence 在 top-1/3/5/10 中的召回，按数据集拆分。若检索上限不足，先处理语料/检索器问题，不把失败归因于 SFT。
2. **答案评分审计**：分层抽查约 200 条 EM=0，检查别名、日期、标点、答案提取和数据集官方 evaluator 差异。

这两项可以与 Full/Teacher 评测并行准备，但不得修改正在运行的 v0 文件。

### P2：量化旧 Teacher 严格过滤造成的数据损失

1. 先盘点旧原始候选、debug 文件、被拒绝问题 ID 和过滤日志是否仍保留，不立即全量重采。
2. 统计答案错误、格式错误、重复查询、重复 information、空结果、无进展等拒绝原因，以及过滤前后 1/2/3+ 搜索轨迹分布。
3. 从“仅因重复或无进展而被拒绝”的问题中分层抽取 1,000–2,000 条做恢复 pilot：第一次重复后允许重新规划；最终成功且答案正确可以保留，一直重复到预算耗尽才拒绝。
4. 比较恢复成功率、多轮占比、Teacher 正确率和搜索成本，确认增加的是有效多跳轨迹，而不是更长的无效轨迹。

决策门：若旧原始状态可恢复、Prompt 兼容且局部补采足以恢复合理的多轮分布，则复用旧有效轨迹并只补采缺失部分；若 Prompt 将实质改变、原始失败状态丢失或旧数据严重偏向一轮轨迹，才进行全量重新生成。

### P2：在新一轮 Teacher 生成前锁定 canonical Prompt

1. v0→v1 期间保持当前评测 Prompt 不变，以便只测终止规则影响。
2. Teacher 恢复 pilot 可以先沿用旧 Prompt，用于量化“严格过滤”单变量影响。
3. 正式补采/重采前，对 Teacher 生成、SFT 数据表示和评测 Prompt 做逐字段对照。默认优先以当前评测 Prompt 为 canonical，反向对齐 Teacher 和 SFT，从而复用 v1/v2 基线。
4. 必须统一标签语义、每轮动作数、错误纠正、重复处理、top-k、文档截断、最大轮数、stop strings、答案格式和消息角色。
5. v2 已保留给上下文策略实验，仍使用当前评测 Prompt。若 canonical Prompt 与当前评测 Prompt 有实质变化，另建 v3；先在固定小样本上重测必要对照，确认收益后再扩展，不能跨版本直接比较。

### P2：v2 累计检索历史上下文

1. v2 继承 v1 的重复查询处理、8 次搜索尝试预算和额外最终回答轮，只改变上下文保留方式。
2. 不再只保留“问题 + 最近一次检索交换”，而是按时间顺序累计所有搜索和 information。
3. vLLM 总上下文仍为 4096 tokens，并为本轮生成固定预留 768 tokens；应用模型 chat template 后的输入 prompt 不得超过 3328 tokens。
4. 超预算时优先从最早轮、低排名文档开始裁剪 evidence 正文，保留问题、所有 query、结构标签、文档 ID/标题和最新 evidence；必要时再裁剪最早的历史 think。
5. 先在固定多跳子集上比较 v1/v2，再决定是否扩展；至少重测所有进入第 3 次 assistant generation、因而会受到历史保留变化影响的样本。

验收：明确累计历史是否提高 HotpotQA、2WikiMultiHopQA、MuSiQue 等多跳数据集表现，并报告上下文截断率和每轮实际 token 使用量。

### P3：重建 SFT 数据并只正式训练 Full SFT

1. 使用真实 27B Teacher 轨迹；按问题 ID 去重并排除评测泄漏，在切分后的固定 train/eval 上构造数据。
2. 硬性拒绝答案错误、不可恢复格式、空查询/空结果和持续无进展；对能够纠正后成功的轨迹按证据覆盖、重复率和成本排序，不因第一次重复直接删除整题。
3. 显式核验 NQ/HotpotQA 来源比例、1/2/3+ 搜索分布、长度和截断分布。
4. SFT loss 继续监督 `<think>`、`<search>`、`<answer>`，mask `<information>`、system/user 和 padding；训练前抽样核对 token labels。
5. 先比较 v1 下当前 Full 与 LoRA。若 Full 明确更好，新数据不再完整训练 LoRA；先做短 Full SFT smoke test，正常后从原始 Qwen3.5-4B Base 独立启动正式 Full SFT。

验收：验证 loss、生成格式和多轮行为正常；新 Full 在同一 canonical 协议下与 Base、Instruct、旧 Full、Teacher 对比。近期到此为止，暂不进入 RL 或大规模超参数搜索。

### 当前明确不做

- 暂不启动 GRPO、DAPO、成本奖励或过程级 advantage。
- 暂不同时调整学习率、epoch、top-k、Prompt 和过滤规则。
- 暂不因平均搜索次数下降就判定模型更好；必须结合 EM/F1、正确样本搜索次数和无效行为分析。

## 阶段 0：环境冻结

锁定 Python、CUDA、PyTorch、Transformers、vLLM、LLaMA-Factory 和 veRL 版本，记录 GPU 型号与显存。现代 Qwen3.5 推理环境与旧版 Search-R1/veRL 环境隔离；每次实验保存 `pip freeze` 和启动命令。

## 阶段 1：数据与检索器

准备 HotpotQA、NQ、Bamboogle（如使用）及 Wikipedia 2018。构建 Lucene BM25，固定 analyzer、索引版本、top-k=3。验证空查询、未命中、排序稳定性、docid 到正文回读和重复查询行为。

验收：索引文档数与校验和固定；健康查询返回可解析正文；检索服务可被生成和评测脚本复用。

## 阶段 2：多轮环境与基线

定义严格序列：assistant 输出 `<think>`，可输出 `<search>query</search>`；环境返回 `<information>...</information>`；最终输出 `<answer>...</answer>`。实现最多 8 轮、查询去重、空结果处理、停止条件和轨迹记录。

先评估 Base 模型，记录 EM/F1、平均搜索次数、零搜索率、无效调用率和失败类型，作为后续所有训练的对照。

## 阶段 3：Teacher 轨迹

用 Qwen3.5-27B 生成候选轨迹。过滤空/重复查询、空证据、格式错误、缺少搜索、答案不匹配和违反工具顺序的样本。保留原始轨迹、过滤原因、数据源和搜索次数；固定 15,000 train + 1,000 eval，禁止 oracle-only 样本混入。

## 阶段 4：SFT 数据与监督定义

将检索结果作为独立环境消息，确保只对 assistant 生成内容计算 loss。对每条样本抽样核对 `input_ids`、`labels`、turn 边界和有效监督 token 比例；检查 truncation 不会截断 answer 或破坏工具序列。LoRA 与 Full SFT 必须使用完全相同的数据、模板、cutoff、seed 和 eval 集，仅改变 finetuning type 与学习率。

建议初始配置：cutoff 4096、有效全局 batch 8、LoRA lr 2e-5、Full lr 1e-5、cosine、warmup 5%、BF16、gradient checkpointing、每 250 step eval/save。若 loss 早期平台，优先检查 mask、截断比例、重复样本和学习率，再调整超参。

验收：训练无异常；eval loss 与生成式 EM 同步改善；checkpoint 可恢复；正式报告只使用正确 mask 的重跑结果。

## 阶段 5：统一评测

使用固定 test.parquet 50k 子集和相同 BM25/生成参数评估 Base、LoRA、Full。保存逐条轨迹、答案、搜索次数和错误分类，计算 EM、F1、均值及置信区间。不得用训练集或 SFT eval 替代测试结果。

## 阶段 6：Outcome GRPO

在 Full/LoRA 中选定正式 SFT checkpoint，先实现仅答案正确性奖励的 GRPO。固定 rollout 数、采样温度、KL、最大轮数和 batch；监控 reward、EM、搜索次数、长度和 KL，防止奖励崩溃。

## 阶段 7：成本感知奖励

加入搜索次数或额外搜索惩罚，比较相同准确率下的成本变化。报告奖励权重、准确率—搜索次数曲线及过早停止、零搜索等失败案例。

## 阶段 8：DAPO 与动态采样

迁移到兼容当前 vLLM 的新版 veRL/DAPO 实现，明确动态采样、裁剪、KL 和长度设置。先做小规模 smoke test，再进行正式训练；记录吞吐、显存、有效 rollout 比例和稳定性。

## 阶段 9：过程级 credit assignment

按搜索结果相关性、后续答案贡献或状态等价类估计中间 advantage，改善多工具调用的 credit assignment。与 outcome-only 对照，做移除过程奖励和移除成本奖励的消融。

## 阶段 10：最终材料

整理主结果、消融、训练曲线、典型成功/失败轨迹、资源成本和可复现实验命令。所有表格引用唯一实验 ID，区分诊断实验、正式实验和任何降级实验。
