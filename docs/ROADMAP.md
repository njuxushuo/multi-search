# 多轮检索问答复现实验方案

## 目标与验收口径

复现 Qwen3.5-27B Teacher → Qwen3.5-4B Student 的多轮 BM25 检索问答链路。正式结果必须包含 15,000 条真实搜索轨迹、独立 1,000 条 SFT eval，以及固定 50k 跨数据集测试子集。核心指标为 EM、F1、平均检索次数和无效工具调用率；所有实验保存配置、代码版本、随机种子、日志和 checkpoint。

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
