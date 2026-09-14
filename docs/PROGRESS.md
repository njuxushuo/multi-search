# 项目进度

更新时间：2026-09-14

修正版多轮 SFT 数据已重建；LoRA 正在 GPU 0–3 上运行，成功后由 watcher 自动启动 full SFT。

## 当前结论

项目已完成数据、Wikipedia 2018、BM25 和 Teacher 轨迹准备。错误的旧 SFT 数据已移入 `archive/invalid_sft_data_deleted-20260914-1208/`；从 Teacher 合并文件重新生成了 15,000 train + 1,000 eval 多轮数据，evidence 保留为非监督 turn。旧 full SFT 已停止，不再作为正式结果。

## 已完成

- Qwen3.5-27B Teacher 轨迹：合并 18,453 条；严格筛选出 15,000 train + 1,000 eval。
- Wikipedia 2018：21,015,324 篇文档；BM25 Lucene 索引构建成功。
- LoRA SFT：1 epoch / 1,875 steps，train loss 1.0806、eval loss 1.0571；因 loss mask 问题作废。
- 已固定统一评测入口：Base、SFT、RL 均使用同一 test.parquet 50k 子集、BM25 top-k=3、最多 8 轮，并输出 EM 与平均检索次数。

## 本次修正

`validate_teacher_traces.py` 现在将每个 `<information>...</information>` 片段改成独立 `human` turn，并标记 `metadata.loss_mask=assistant_turns_only_v2`。train/eval 结构校对通过，所有 evidence 保留且不在 gpt turn 中。

修正版 LoRA 实验 `qwen35-4b-lora-multiturn-policy-mask` 已启动，SwanLab run 为 `pxhoz0v5`；watcher 将在 LoRA 成功后启动 `qwen35-4b-full-multiturn-policy-mask`。

正式重跑前必须抽样检查 token labels：所有 evidence span 的 label 应为 `-100`，assistant 的 think/search/answer token 应保留有效 label；同时确认 eval 使用同一 mask 规则。

## 当前待办

1. 用修正数据重跑 LoRA，记录有效 label token 比例、loss 和生成质量。
2. 重跑 Full SFT，并保存最佳 checkpoint 与配置快照。
3. 完成 Base/SFT 50k 评测，写入 `RESULTS.md`。
4. 在 SFT 结果稳定后实现 outcome GRPO → 成本奖励 → DAPO → 过程级 credit assignment。

## 产物

- 数据：`data/processed/search_sft_qwen35_4b/`
- LoRA（诊断结果）：`outputs/sft_qwen35_4b_lora/`
- Full SFT 日志/checkpoint：`outputs/sft_qwen35_4b_full/`
