# 项目当前状态

更新日期：2026-09-16

本文档只记录“现在什么是真的”。已冻结的评测口径见 `EXPERIMENT_PROTOCOL.md`，正式数字见 `RESULTS.md`，下一步执行顺序见 `ROADMAP.md`。Git 历史保留了旧的日报和初始化记录，不再另行维护容易过时的时间线文档。

## 一句话结论

项目已打通 Teacher 采样、SFT 训练和 50k 端到端评测，且 Full SFT 明显优于 Base 和 LoRA；但当前 15k SFT 数据中 72.05% 只有一次检索，Teacher 生成、SFT 输入和端到端评测又存在 Prompt/上下文策略不一致。因此当前不具备开始 RL 的条件，应先重建更合理的 Teacher 数据并重训一个 Full SFT。

已确定的下一轮模型选择：Teacher 继续使用官方 post-trained `Qwen3.5-27B`；Student 不再从 `Qwen3.5-4B-Base` 启动，改为从官方 post-trained `Qwen3.5-4B` 启动 Full SFT。官方模型名没有 `-Instruct`，文档中的 instruction Student 均指这个 post-trained checkpoint。不再进行 Base/post-trained Student 初始化对照。

新 Teacher/SFT 主线的设计协议已冻结为 `v3 Search-R1-aligned`，详细规范见 `EXPERIMENT_PROTOCOL.md`：使用 parquet 原始 Prompt、连续累计轨迹、BM25 top-k=3、最多 4 次 search + 1 次 answer-only、总上下文/SFT cutoff 8192、单次生成 768、单次 top-3 observation 768。当前状态是“设计冻结、代码和 pilot 待实现”，不能误记为已经完成。

## 已完成与当前运行项

| 模块 | 状态 | 已核验事实 |
|---|---|---|
| Wikipedia 2018 | 已完成 | `data/corpus/wiki-18.jsonl`，21,015,324 篇文档 |
| Lucene BM25 | 已完成 | `data/index/bm25`，HTTP 检索服务支持 top-k=3 |
| 27B Teacher 权重 | 已完成 | `models/Qwen3.5-27B`，11 个 safetensors 分片已校验 |
| 第一版 Teacher 轨迹 | 已完成 | 4 个 shard 合并后 18,453 条严格有效轨迹；合并时拒绝 1,115 条、去重 168 条 |
| 第一版 SFT 数据 | 已完成 | 15,000 train + 1,000 teacher-forced eval，evidence 不计 loss |
| LoRA SFT | 已完成 | 正确 mask 版 1 epoch / 1,875 steps；后续不再作为必经阶段 |
| Full SFT | 已完成 | 正确 mask 版 1 epoch / 1,875 steps；当前最强的本地 4B 模型 |
| v0/v1 50k 评测 | 已完成 | Base、LoRA、Full SFT 已严格合并；正式结果见 `RESULTS.md` |
| 27B Teacher 评测 | 外部运行 | 由其他服务器执行，本机不自动启动 |
| Qwen3.5-4B 官方后训练版 v1 | 运行中 | 模型路径 `/data0/xs/models/Qwen3.5-4B`；官方名称没有 `-Instruct`，与 Base 权重不同；使用 GPU 0–3 数据并行评测，未完成前不写入正式结果 |
| RL | 未启动 | 数据与协议尚未达到准入门槛；老 Search-R1/veRL 也不兼容当前 Qwen3.5 推理栈 |

## 当前 SFT 数据快照

正式数据位于 `data/processed/search_sft_qwen35_4b/`，来自 `data/processed/teacher_traces_qwen35_27b.merged.jsonl`。

| 切分 | HotpotQA | NQ | 1 次检索 | 2 次 | 3+ 次 | 总数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 10,827 | 4,173 | 10,808 (72.05%) | 3,230 (21.53%) | 962 (6.41%) | 15,000 |
| eval | 757 | 243 | 717 (71.70%) | 219 (21.90%) | 64 (6.40%) | 1,000 |

train 的平均检索次数为 1.373。这个分布能教会基本工具格式，却不足以充分教会多跳搜索规划，也不宜直接作为后续“降低检索成本”RL 的初始策略。

### 已确认的数据问题

1. Teacher 实时生成使用完整工具说明 Prompt，而 SFT 转换后的首个 `human` turn 只有原始问题，存在 Prompt 不一致。
2. Teacher 实时生成和 v1 评测只保留“问题 + 最近一轮有效检索”，SFT 序列却保留全部历史 evidence，存在上下文策略不一致。
3. Teacher 遇到第一次重复查询就终止；严格 validator 又会拒绝整条含重复 query 或重复 information 的轨迹，对长轨迹有明显选择偏差。
4. 老轮次约 98,651 个候选中，最终合并得到 18,453 条。启动器未传 `debug_output`，大多失败候选的完整状态和拒绝原因没有保存，无法对旧轮次做完整离线恢复。
5. 现有 15k/1k 是校验后记录的前 15k 和后 1k，没有基于 question ID 的确定性分层打乱，也没有在 SFT metadata 中保留 question ID/gold answers。
6. 当前 1k SFT eval 全部是 Teacher 已经答对的成功轨迹，适合 teacher-forced loss，不适合单独担任端到端 EM 验证集。

## 已完成的 SFT 快照

| 模型 | 训练 | train loss | 最终 eval loss | 主要配置 |
|---|---:|---:|---:|---|
| LoRA | 1 epoch / 1,875 steps | 0.31459 | 0.29869 | rank 16, alpha 32, all linear modules, lr 2e-5 |
| Full SFT | 1 epoch / 1,875 steps | 0.23753 | 0.30743 | ZeRO-3, lr 1e-5 |

两者都使用 Qwen3.5-4B Base、BF16、cutoff 4096、有效全局 batch 8、cosine scheduler、warmup 5%。Full SFT 的 eval loss 在 step 250/500/750/1000/1250/1500/1750/1875 为 0.38493/0.37344/0.36251/0.34193/0.32629/0.31313/0.30777/0.30743，至 epoch 结束仍在改善。LoRA 的当次 trainer state 只留下最终 eval，不应追溯声称它完成了每 250 step 验证。

Loss 口径：只监督 assistant 生成的 `<think>`、`<search>` 和 `<answer>` token；`<information>`、system/user 以及 padding 的 label 为 `-100`。这是合理且应继续保留的方案。

## 当前可信的评测结论

- 非空修正口径下，v1 总体 EM：Base 18.496%、LoRA 24.370%、Full SFT 28.792%。
- Full SFT 比 Base 提高 10.296 个百分点，比 LoRA 提高 4.422 个百分点，而 Full/LoRA 的平均实际 BM25 次数接近（2.038 vs 2.019）。因此下一轮数据实验只需正式训练 Full SFT。
- Full SFT v1 的 2WikiMultiHopQA/MuSiQue EM 只有 9.680%/7.774%，而重复 query 和搜索预算耗尽仍很常见。现在的主要问题是多跳策略与证据利用，不是立即惩罚搜索次数。
- v0/v1 的旧 online scorer 有“空预测与归一化后为空的 gold 相等”边界错误。`RESULTS.md` 中的正式数字已离线按非空口径修正；新评测器必须在线直接修复。

## 环境与已知兼容性

- Qwen3.5 生成/评测使用 `search-r1` Conda 环境：Python 3.10、PyTorch 2.10.0+cu128、Transformers 4.57.6、vLLM 0.18.0。
- Qwen3.5 在本机需使用 `gdn_prefill_backend=triton`；FlashInfer GDN prefill 在当前 CUDA toolkit 下会触发 PTX 编译问题。27B Teacher 还需 eager mode 规避 torch.compile/FakeTensorMode 问题。
- SFT 实际使用本地 `/data0/xs/LLaMA-Factory/src`，启动入口为 `scripts/run_sft_qwen35_4b_{lora,full}.sh`。
- `third_party/Search-R1` 内置的 veRL v0.1 只适配 vLLM 0.3–0.6，不能直接用于当前 Qwen3.5 + vLLM 0.18 RL。未来 RL 需迁移到新版上游 veRL，不应在旧 vendored 实现上强行打补丁。

## 关键产物索引

| 产物 | 路径 |
|---|---|
| Teacher 合并轨迹 | `data/processed/teacher_traces_qwen35_27b.merged.jsonl` |
| 现有 SFT train/eval | `data/processed/search_sft_qwen35_4b/{train,eval}.jsonl` |
| LoRA 及合并模型 | `outputs/sft_qwen35_4b_lora_multiturn_masked{,_merged}` |
| Full SFT | `outputs/sft_qwen35_4b_full_multiturn_masked` |
| v0 评测 | `outputs/eval/qwen35_4b_{base,lora,full}_50k_docnovelty.*` |
| v1 评测 | `outputs/eval/v1/qwen35_4b_{base,lora,full}_v1_neutral_50k.*` |
| 后训练 4B 运行中分片 | `outputs/eval/v1/qwen35_4b_posttrained_v1_neutral_50k.shard*.jsonl` |
| SwanLab 项目 | `search-r1` |

## 文档维护规则

1. 只有已严格合并、按正式 scorer 重算的完整结果才写入 `RESULTS.md`。
2. 运行中进度不持续追加到文档；使用 shard 行数、日志和 GPU 进程实时查看。
3. 已冻结的 v0/v1 协议只做事实性更正，不修改其实验定义。
4. 新 Prompt、上下文策略或数据版本必须使用新实验 ID 和新输出目录，不覆盖旧产物。
