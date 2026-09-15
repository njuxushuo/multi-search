# 项目进度

更新时间：2026-09-15

## 2026-09-15 更新

- 当前第一版统一评测定义为 v0，完整协议已冻结在 `docs/EXPERIMENT_PROTOCOL.md`。
- Base、LoRA、Full SFT 的 v0 50k 评测均已完成并严格合并；Full SFT 最终 EM 27.892%、F1 33.134%、平均成功检索 1.815 次、平均搜索尝试 2.266 次。27B Teacher 已在其他服务器评测，本机顺序脚本默认不再启动 Teacher（仅显式设置 `EVAL_RUN_TEACHER=1` 时启动）。
- Base、LoRA、Full SFT 的 v0 七数据集分项 EM、平均成功检索次数和平均搜索尝试次数已写入 `docs/RESULTS.md`；三个模型的分组样本数均核对为 50,000。Teacher 完整结果待从另一服务器回传后按相同表结构补录。
- v1 已按“重复查询不调用 BM25、返回中性 no-progress observation、最多 8 次搜索尝试后追加一次 answer-only generation”实现为独立 evaluator；v0 文件未修改，v1 输出固定隔离在 `outputs/eval/v1/`，完成核验后待启动。
- 近期实验顺序已整理到 `docs/ROADMAP.md` 的“近期实验计划”：先完成 v0 四模型基线，再局部修复重复搜索提前终止并加入 Qwen3.5-4B-Instruct；随后审计旧 Teacher 过滤损失，完成 Prompt 对齐、数据重建和新 Full SFT。RL 暂缓。
- v0 结果不可覆盖；v1 只修正重复查询立即终止，并在最多 8 次搜索尝试后额外提供一次最终回答生成；v2 继承 v1，再改为累计保留检索历史并按 3328-token 输入预算确定性裁剪。

## 2026-09-14 历史记录

修正版多轮 SFT 数据已重建；LoRA 正在 GPU 0–3 上运行，成功后由 watcher 自动启动 full SFT。

### 当日结论

项目已完成数据、Wikipedia 2018、BM25 和 Teacher 轨迹准备。错误的旧 SFT 数据已移入 `archive/invalid_sft_data_deleted-20260914-1208/`；从 Teacher 合并文件重新生成了 15,000 train + 1,000 eval 多轮数据，evidence 保留为非监督 turn。旧 full SFT 已停止，不再作为正式结果。

### 当日已完成

- Qwen3.5-27B Teacher 轨迹：合并 18,453 条；严格筛选出 15,000 train + 1,000 eval。
- Wikipedia 2018：21,015,324 篇文档；BM25 Lucene 索引构建成功。
- LoRA SFT：1 epoch / 1,875 steps，train loss 1.0806、eval loss 1.0571；因 loss mask 问题作废。
- 已固定统一评测入口：Base、SFT、RL 均使用同一 test.parquet 50k 子集、BM25 top-k=3、最多 8 轮，并输出 EM 与平均检索次数。

### 当日修正

`validate_teacher_traces.py` 现在将每个 `<information>...</information>` 片段改成独立 `human` turn，并标记 `metadata.loss_mask=assistant_turns_only_v2`。train/eval 结构校对通过，所有 evidence 保留且不在 gpt turn 中。

修正版 LoRA 实验 `qwen35-4b-lora-multiturn-policy-mask` 已启动，SwanLab run 为 `pxhoz0v5`；watcher 将在 LoRA 成功后启动 `qwen35-4b-full-multiturn-policy-mask`。

正式重跑前必须抽样检查 token labels：所有 evidence span 的 label 应为 `-100`，assistant 的 think/search/answer token 应保留有效 label；同时确认 eval 使用同一 mask 规则。

### 当日待办（已由 2026-09-15 计划取代）

1. 用修正数据重跑 LoRA，记录有效 label token 比例、loss 和生成质量。
2. 重跑 Full SFT，并保存最佳 checkpoint 与配置快照。
3. 完成 Base/SFT 50k 评测，写入 `RESULTS.md`。
4. 在 SFT 结果稳定后实现 outcome GRPO → 成本奖励 → DAPO → 过程级 credit assignment。

### 当日产物

- 数据：`data/processed/search_sft_qwen35_4b/`
- LoRA（诊断结果）：`outputs/sft_qwen35_4b_lora/`
- Full SFT 日志/checkpoint：`outputs/sft_qwen35_4b_full/`
