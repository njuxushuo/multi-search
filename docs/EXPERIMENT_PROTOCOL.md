# 实验记录规范

## 文档职责

- 本文件保存已经冻结、可复现的实验协议及指标口径。
- 当前事实快照保存在 `docs/PROJECT_STATUS.md`，近期执行顺序保存在 `docs/ROADMAP.md`。
- 完成后的正式数字写入 `docs/RESULTS.md`，不把运行中的中间值当作最终结果。
- v0/v1 是历史基线；v2 是只改变历史上下文保留方式的预定消融；新 Teacher/SFT 主线使用本文件定义的 **v3 Search-R1-aligned canonical 协议**。v3 的设计已冻结，但在代码、单元测试和 pilot 验收完成前不得写成“已实现”。

每次运行前复制一份配置并分配唯一实验 ID。必须记录：

- Git commit、运行日期、随机种子
- GPU 型号、数量、显存
- Python/CUDA/PyTorch/Transformers/veRL 版本
- 模型名称和精度（BF16/FP16/量化）
- 数据集版本、样本数和过滤规则
- 检索语料版本、索引参数、top-k
- batch size、学习率、最大轮数和最大输出长度
- 完整启动命令、日志和 checkpoint 路径
- EM、F1、平均检索次数、无效调用率

结果应先写入 `RESULTS.md`，原始日志和轨迹保存在项目外部或约定目录，并在表格中留下路径。

## 协议版本注册表

| 对话简称 | 机器 ID | 含义 |
|---|---|---|
| v0 | `qwen35_search_r1_v4_dp_doc_novelty` | 第一版历史评测；重复 query 立即终止、8 次搜索、latest exchange |
| v1 | `qwen35_search_eval_v1_neutral_repeat_final_answer` | 历史修正版；重复 query 中性恢复、8 次搜索加回答轮、latest exchange |
| v2 | 尚未运行 | 只对 v1 增加累计历史/token-budget 裁剪的预定消融 |
| **R3.0** | **`searchqa_repro_v3_0_0`** | 下一轮完整复现实验：27B Teacher 数据、筛选、4B post-trained Full SFT、dev/final eval 的统一方案 |

后续对话中单独提到“R3.0”，即指本文件整个 R3.0 章节以及机器配置 `configs/protocols/searchqa_repro_v3_0_0.json`，不只是评测状态机。Prompt、chat template 语义、top-k、observation 格式、轮数/token budget、筛选配额、loss mask、模型初始化和主指标中任何一项发生语义变化，都必须分配新版本，不能仍称 R3.0。

## v0 冻结评测协议（2026-09-15）

### 协议身份与用途

项目文档中的 **v0** 指当前正在使用的旧终止规则评测。代码写入结果 metadata 的内部协议名为 `qwen35_search_r1_v4_dp_doc_novelty`。v0 用于保存 Base、LoRA、Full SFT、27B Teacher 的第一版可比基线；后续修正重复搜索处理时必须产生新协议版本，不能覆盖 v0 文件。

截至 2026-09-16，Base、LoRA 和 Full SFT 的 v0/v1 本地评测均已完成并严格合并。Teacher 已改在其他服务器评测，本机顺序启动脚本不再启动 Teacher。官方后训练模型 `Qwen3.5-4B`（官方命名没有 `-Instruct`）不属于 v0 四模型集合，当前直接使用修正后的 v1 协议评测。

### 代码与运行环境

| 项目 | v0 固定值 |
|---|---|
| Git HEAD | `479d3f3e2581e45dac8eef645f4a29a0175e9242` |
| 工作树状态 | dirty；评测脚本在当前工作树中尚未全部纳入 Git，因此必须同时核对下列文件 SHA-256 |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.6 |
| vLLM | 0.18.0 |
| GPU | GPU 0–3，各一张 NVIDIA H800 PCIe 81559 MiB |
| 并行方式 | 一个模型占四卡；4 个 TP=1 数据并行 worker；模型之间顺序评测 |
| 模型精度 | BF16 |
| vLLM | `max_model_len=4096`、`gpu_memory_utilization=0.90`、`enforce_eager=True`、`gdn_prefill_backend=triton` |

关键文件 SHA-256：

```text
fdf21f5a2d6b62b4a853a3ff25b75cd28aa2fee62b93d828afa46fa032314e3c  scripts/evaluate_qwen35_search.py
70952854525bdbc88331ac2448f9f912326364b764bbe303598d90fca8f6c6ac  scripts/run_eval_qwen35_search.sh
0b03dbfa34ec0e1415322a37b3baa41af7b0842c5f55523c77d9d276071fd63f  scripts/run_eval_qwen35_search_4gpu_sequential.sh
ff36204d722bf4b8598a57a9b5199c9b34c027cbeca7c5c2020e415d202c07a3  scripts/merge_eval_shards.py
```

### 模型集合

| 名称 | v0 实际加载路径 |
|---|---|
| Base | `/data3/xs/models/Qwen3.5-4B-Base` |
| LoRA | `/data0/xs/search/outputs/sft_qwen35_4b_lora_multiturn_masked_merged` |
| Full SFT | `/data0/xs/search/outputs/sft_qwen35_4b_full_multiturn_masked` |
| Teacher | `/data0/xs/search/models/Qwen3.5-27B` |

LoRA 在 v0 中评测的是已经合并权重的模型目录，并非运行时挂载 adapter。

### 测试集与抽样清单

- 源文件：`data/processed/nq_hotpotqa_train/test.parquet`
- 固定清单：`data/processed/search_eval/searchqa_50k_seed42_hnb_first.jsonl`
- 样本数：50,000
- 抽样随机种子：42
- evaluator canonical selection SHA-256：`9f1112319c9c5d712decd9ffa4f396ed69420f666e1dcb88bb2f5665f75fc4c4`
- manifest 文件 SHA-256：`d441ee70a38bcd7136b1ad4e0f5afc79f979753582d5f1020a525291d16edc09`
- manifest 创建方式：对源测试集以 seed 42 打乱，选择前 50,000 条，再将 HotpotQA、NQ、Bamboogle 排在前面；实际样本身份由 manifest 固定，不再重新抽样。

| 数据集 | 条目数 |
|---|---:|
| HotpotQA | 7,158 |
| NQ | 3,505 |
| Bamboogle | 122 |
| TriviaQA | 10,950 |
| PopQA | 13,775 |
| 2WikiMultiHopQA | 12,149 |
| MuSiQue | 2,341 |
| 合计 | 50,000 |

已知 metadata 限制：worker/合并后的 summary 中 `source_order` 为空，因为 `--source-order` 只传给了 manifest 准备步骤；这不改变固定 manifest 的样本和顺序。

### 检索环境

| 项目 | v0 固定值 |
|---|---|
| 语料 | Wikipedia 2018，`data/corpus/wiki-18.jsonl`，21,015,324 篇文档 |
| 检索器 | 本地 Lucene BM25 HTTP 服务 |
| 索引 | `data/index/bm25` |
| 索引 commit 文件 | `segments_1` SHA-256 `44724d6637058e5ed1793b859830e3fc15f1951c2ae758e411544251a61a034c` |
| 服务 | `http://127.0.0.1:8000/retrieve` |
| 每次返回 | top-k=3 |
| 单文档内容截断 | 前 1,800 个 Python 字符 |
| 信息拼接 | 3 篇命中文档以空行连接，包裹在 `<information>...</information>` 中 |

v0 不对命中文档做二次排序或压缩。一次成功查询不保证一定得到 3 篇非空文档，因此“搜索次数”和“返回文档数”是两个独立指标。

### Prompt 与消息模板

每道题初始只有一个 `user` 消息，随后通过模型自身 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)` 渲染。v0 的精确任务文字为：

```text
Answer the question using the search tool when needed. Reason inside <think>...</think>. To search, emit exactly <search>query</search>; after each search you will receive <information>...</information>. Finish with exactly <answer>final answer</answer>.
Question: {question}
```

一次成功搜索后追加一个 assistant 搜索消息和一个 user 证据消息：

```text
<information>
{top-k passages}
</information>
Use the evidence above. If the evidence answers the question, stop searching and output the required answer tag.
```

达到最后一次允许的成功搜索时，末句替换为：

```text
This is the final allowed search; do not search again. Output the required answer tag now.
```

上下文保留策略也是 v0 的组成部分：当消息数超过 3 时，仅保留初始问题和最近一次 assistant/search + user/information 交换。完整轨迹会写入结果文件，但更早的检索证据不会继续留在模型实时上下文中。

### 生成参数与动作语法

| 参数 | v0 固定值 |
|---|---:|
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| 每轮 `max_tokens` | 768 |
| stop strings | `</search>`、`</answer>`，并保留 stop string |
| 最大 assistant 生成轮数 | 8 |
| 最大成功搜索次数 | 8 |
| batch size | 每个 worker 32 |
| seed | 42 |

合法动作必须严格匹配以下二选一，动作外不能有额外文本：

```text
<think>...</think><search>query</search>
<think>...</think><answer>final answer</answer>
```

若 Qwen 只输出 `</think>` 而省略隐式 `<think>`，评测器会在解析前补上 `<think>`。格式不符合严格语法时立即以 `format_error` 终止；如果文本中仍能提取 `<answer>`，该答案仍参与 EM/F1，但轨迹不再属于 valid trajectory。

### v0 停止和重复处理规则

1. 生成合法 `<answer>`：正常结束。
2. 空查询：立即以 `invalid_search` 结束。
3. 当前查询经小写、去标点、去英文冠词并压缩空白后，与历史查询相同：不再调用 BM25，立即以 `repeated_search` 结束。
4. 查询不同但返回的全部文档都已出现：记录 `no_new_documents`、`redundant_search_count += 1` 和 no-progress，但 **不会当场结束**；模型可以继续下一轮。
5. 检索异常或空结果：立即结束。
6. 超出搜索上限：立即结束。
7. 8 轮内没有答案：记录 `max_rounds_without_answer`。

因此，v0 的核心已知缺陷是“重复查询立即停止”，而不是“相同文档立即停止”。v1 的局部续测应只改变重复/可恢复错误的处理，其他参数保持不变。

### 答案评分

所有 50,000 条都计入分母，不进行拒绝采样。答案归一化步骤为：

1. 转小写；
2. 删除 Python `string.punctuation` 中的 ASCII 标点；
3. 删除英文冠词 `a`、`an`、`the`；
4. 合并多余空白。

EM 为预测与任一 `golden_answers` 归一化后完全相同。F1 为预测与各 gold 的 token-level F1 最大值。当前 summary 的总体 EM/F1 是按 50,000 条样本直接计算的 **micro average**；同时输出每个数据集结果，但尚未输出七数据集等权 macro average。

#### 2026-09-16 非空评分修正

评测完成后确认旧实现存在空字符串边界问题：空 prediction 会与归一化后为空的参考别名（例如 `!!!`、`---`、`The A`）相等，并被误记为 EM=1；`token_f1()` 也会把两侧空 token 列表记为 1。正式报告增加以下约束：

1. prediction 归一化后为空时，EM=0、F1=0；
2. 单个 gold 归一化后为空时，不参与该题的 EM/F1 最大值计算；
3. 若所有 gold 均归一化为空，该题不能因空预测获得正确分；
4. 已生成的原始 JSONL 和 summary 保持不变，通过逐条预测离线重算得到正式数字。

该修正同时应用于 v0/v1，属于确定性的评分审计，不改变任何模型生成、状态机、检索行为或样本身份。`docs/RESULTS.md` 记录原始 summary 与修正值；以后新 evaluator 应在在线评分阶段直接执行上述非空检查。

置信区间口径：EM 使用 95% Wilson interval；F1 和平均成功搜索次数使用正态近似的 95% mean CI。

### 搜索与轨迹指标口径

- `search_count`：成功返回至少一个非空 hit 的 BM25 调用数。
- `search_attempt_count`：模型提出搜索动作的次数；包含空查询、重复查询和达到上限后的尝试。
- `progressive_search_count`：本轮至少包含一个此前未见文档的成功搜索数。
- `redundant_search_count`：成功检索但没有任何新文档的搜索数。
- `retrieved_document_count`：所有成功搜索实际返回的文档条目总数。
- `unique_document_count`：按 doc ID；缺失 ID 时按内容哈希去重后的文档数。
- `mean_searches`：全体样本 `search_count` 的均值，不是“正确样本平均搜索次数”。
- `format_compliance_rate`：所有生成动作始终满足严格动作语法的样本比例。
- `valid_trajectory_rate`：存在答案、格式合规且没有任何 invalid reason 的样本比例。
- `strict_success_rate`：EM=1 且 valid trajectory。
- `recovered_success_rate`：EM=1、存在答案且曾发生 no-progress。

重要限制：当前 `invalid_tool_call_rate` 实际定义为“轨迹是否出现任意 invalid reason”，会纳入重复查询、无新增文档、检索异常和超轮次等情况；它不是纯粹的工具调用格式错误率。后续报告必须保留这个说明，v1 可增加更细的互斥指标，但不能静默改变 v0 数字含义。

### 落盘、恢复与合并

- `eval_id % 4` 决定四个 shard 的分配，每个 shard 恰有 12,500 条。
- 每完成一个最多 32 条的 batch，逐条追加 JSONL，随后 `flush + fsync`。
- `--resume` 按已落盘 `eval_id` 跳过完成样本。
- 四个 shard 完成后，合并脚本检查 metadata、shard index、重复 ID、缺失 ID和额外 ID；只有恰好覆盖固定 manifest 的 50,000 个 ID 才产生合并文件和 summary。
- v0 输出前缀为 `outputs/eval/qwen35_{4b_base,4b_lora,4b_full,27b_teacher}_50k_docnovelty`。

### v0 与后续版本的可比性边界

v1 计划保持模型、manifest、Prompt、BM25、top-k、文档截断、解码参数、评分器和“只保留最近一次检索交换”的上下文策略不变，只修正重复查询立即终止，以及为搜索预算耗尽的轨迹预留一次最终回答机会。v1 结果必须另存，并报告受影响样本数、恢复正确数、全量 EM 增量和额外搜索成本。

v2 已专门保留给“累计检索历史并按 token budget 裁剪”。若后续还改变 Prompt、top-k、答案归一化或模型 chat template 语义，则应建立 v3 或独立实验分支，并在同一新协议下重新获得必要的对照模型结果，不能与 v0/v1/v2 直接混为同一实验表。

## 已确定的版本演进

| 版本 | 重复查询 | 搜索预算后的回答机会 | 检索历史上下文 | 用途 |
|---|---|---|---|---|
| v0 | 归一化查询重复后立即终止 | 无；连续 8 轮搜索后直接结束 | 原始问题 + 最近一次检索交换 | 冻结的四模型旧协议基线 |
| v1 | 不立即终止，不重复调用 BM25，返回中性 no-progress observation | 有；最多 8 次搜索尝试后保留一次最终回答生成 | 原问题 + 最近一次有效检索；纠错生成时临时附带当前重复动作和中性 observation | 隔离评估终止规则的影响，并补充官方后训练 4B 基线 |
| v2 | 继承 v1 | 继承 v1 | 累计保留全部检索交换，超过 prompt token budget 时确定性裁剪 | 隔离评估多跳历史证据保留的影响 |

### v1 冻结实现协议：重复查询与最终回答机会

v1 不是修改 v0 文件，而是新的评测协议和输出命名空间。除以下两点外，v0 参数保持不变。

实现协议名为 `qwen35_search_eval_v1_neutral_repeat_final_answer`。首次正式运行前的关键文件 SHA-256：

```text
9ccb732f4835bf435d8a5dd580b04e7f00ea1079f75b11c00078f07356c0299c  scripts/eval_protocol_v1.py
ebea9564d302a56d9408a5c1ca0a9cf28f9e967ce4ad71869022dcbb8cac040b  scripts/evaluate_qwen35_search_v1.py
4551e83fc85bcb93bfbd059b1188fc790b84d9b510a0dc6495831cd29ad46dbf  scripts/merge_eval_shards_v1.py
912b2251795842f3cbd7727a58446714492c9fb19d6a6bb6610c0faa527e843c  scripts/run_eval_qwen35_search_v1.sh
9ce43c92f2fa2480c76c9b6aba84ebcf76d3c3bc6cd8e62d300464d500225656  scripts/run_eval_qwen35_search_v1_4gpu.sh
```

#### 1. 重复查询不再立即结束

- 每次模型输出合法 `<search>` 都使 `search_attempt_count += 1`，包括重复查询。
- 查询经与 v0 相同的 `normalize()` 后若与历史查询相同，不再次调用 BM25，不增加 `search_count`。
- 环境追加固定、与 gold 无关且不使用“错误/重复”等措辞的中性 no-progress observation。
- 该次重复仍消耗一次搜索尝试预算，记录 `repeated_search_query` 和 no-progress，但轨迹继续。
- 不同查询返回相同文档时仍执行 BM25，增加成功搜索和 redundant/no-progress 计数，并继续运行。

固定 observation 如下，实施时若修改文字必须更新协议版本或脚本哈希：

```text
<information>
No new information was retrieved. Use the available evidence to answer, or issue a different search query.
</information>
```

为避免短窗口在纠错时把最近证据挤掉，下一次生成临时使用“原始问题 + 最近一次有效 search/information + 当前重复 search + 中性 observation”。如果模型随后产生新的有效搜索，窗口立即恢复为“原始问题 + 最新一次有效 search/information”。这不累计更早检索历史，因此不引入 v2 的变量。

#### 2. 为最终答案预留一次生成

- 搜索尝试预算固定为 8；`search_attempt_count` 包含成功、重复、空查询以及其他合法 `<search>` 尝试。
- `search_count` 仍只统计成功返回非空结果的 BM25 调用，因此始终满足 `search_count <= search_attempt_count <= 8`。
- 模型可以在任意轮提前输出 `<answer>` 并结束。
- 第 8 次搜索尝试处理完毕后，环境明确通知搜索预算已耗尽，并额外提供一次 final-answer generation。
- 最终回答轮只接受合法 `<answer>`；若模型再次输出 `<search>`，不执行检索，以 `search_limit` 结束。
- 因此 v1 最多有 8 次搜索尝试和 1 次保留回答生成，即最多 9 次 assistant generation。第 9 次不是新的搜索额度。

若第 8 次尝试成功，先返回实际 `<information>`，再追加“搜索预算已耗尽、现在回答”的指令；若第 8 次尝试是重复查询，则返回中性的预算耗尽 observation，要求使用已有证据回答。空查询、空检索结果、检索异常和格式错误仍是终止性错误，不获得额外恢复轮。

#### v1 局部重测边界

- v0 中 `repeated_search` 结束的样本：从保存的停止状态继续；若无法恢复状态，使用相同确定性解码从该 ID 重跑并单独标记。
- v0 中 `max_rounds` 且已完成第 8 次搜索的样本：直接从第 8 次信息后的状态补一次回答生成。
- 未触发上述行为且输出不受状态机变化影响的样本可以原样复用。
- 普通 `format_error`、合法但错误的 `<answer>`、空检索结果和检索服务错误仍按 v0 停止；v1 不向模型反馈答案是否正确，也不使用 gold 决定是否继续。
- 合并结果必须保存 `origin=v0_reused`、`origin=v1_continued` 或 `origin=v1_rerun`，以便审计。
- v0 evaluator 和输出保持只读；v1 使用独立入口 `scripts/evaluate_qwen35_search_v1.py`、独立合并器和 `outputs/eval/v1/` 命名空间。未来 v2 必须使用另一入口和 `outputs/eval/v2/`，不得覆盖 v1。

v1 至少报告：被续测 ID 数、重复查询恢复率、8 次搜索后回答率、恢复正确数、全量 EM/F1 增量、成功搜索和搜索尝试增量。可恢复的重复/无新文档与格式错误、检索错误等终止性错误分开记录；端到端 EM/F1 仍对全部题目计分。

### v2 预定协议：累计历史与 token budget

v2 继承 v1 的重复查询、搜索预算和最终回答规则。v2 只研究上下文保留策略，不同时修改 Prompt、BM25、top-k、解码参数或评分器；当前 v0/v1 Prompt 在 v2 中保持不变。

#### 上下文预算

- vLLM `max_model_len` 保持 4096 tokens。
- 每轮最大生成保持 768 tokens。
- 为生成固定预留 768 tokens，因此渲染后的输入 prompt budget 为 `4096 - 768 = 3328` tokens。
- token 数必须在应用该模型自己的 chat template 后，用实际 tokenizer 计算；不能再用字符数近似 token 数。
- 单篇检索结果的 1,800 字符预截断暂时保留，随后再应用全局 3,328-token prompt budget。

#### 历史保留和确定性裁剪顺序

1. 构造完整的按时间顺序消息：初始问题、每轮 assistant 的 `<think>/<search>`、对应 `<information>`，不再直接删除早期整轮消息。
2. 始终保留初始任务说明、原始问题、所有搜索 query、每轮结构标签以及最新一次检索信息。
3. 若渲染后不超过 3,328 tokens，完整保留全部历史。
4. 若超过预算，只裁剪 `<information>` 内的文档正文，优先保留最新轮证据；从最早轮、低排名文档开始确定性缩短。
5. 即使正文被裁剪，也保留该轮 query、文档 ID/标题和明确的截断标记，使模型知道该搜索已经发生。
6. 若仅裁剪 evidence 仍无法满足预算，再从最早的历史 `<think>` 正文开始裁剪；不得删除原始问题、搜索 query、最新 evidence 或结构边界。
7. 最终传入 vLLM 前再次精确计数并断言 `prompt_tokens <= 3328`；违反断言的样本不得静默运行。

v2 必须新增逐样本字段：`prompt_token_count_by_round`、`evidence_tokens_before_truncation`、`evidence_tokens_after_truncation`、`truncated_rounds` 和 `context_truncated`。

#### v2 对比边界

- v2 与 v1 的唯一核心变量是历史上下文策略。
- 至少重测所有进入第 3 次 assistant generation 的样本，因为从这一时刻起 v0/v1 会删除第一轮检索，而 v2 不会。
- 只进行 0 或 1 次搜索且生成路径不受上下文变化影响的样本可以复用，但合并时同样记录 origin；若资源允许，正式主结果优先全量重跑。
- v2 结果重点比较多跳数据集 EM/F1、2+ 搜索样本 EM、正确样本平均搜索次数、上下文截断率和 evidence token 使用量。

## R3.0 Search-R1-aligned canonical 协议（设计冻结，实现待 pilot）

### 目标与证据优先级

R3.0 用于下一轮 27B Teacher 蒸馏、4B post-trained Full SFT 和同协议端到端评测。它不是 v1/v2 的覆盖更新，而是独立版本；旧结果和输出保持只读。所有无法从简历直接确认的细节按以下顺序确定：

1. 简历明确描述；
2. 本仓库固定数据中的原始字段；
3. vendored Search-R1 上游代码和默认配置；
4. 为避免人为失败而做的最小、显式扩展。

已冻结的核心选择如下：

| 项目 | v3 固定设计 |
|---|---|
| Teacher | 官方 post-trained `Qwen3.5-27B` |
| Student 起点 | 官方 post-trained `Qwen3.5-4B`，不是 `Qwen3.5-4B-Base` |
| 检索 | Wikipedia 2018 + Lucene BM25，top-k=3 |
| 最大搜索 action | 4；之后额外允许一次 answer-only generation |
| 上下文 | 累计的单一连续轨迹，不再只保留最近一次交换 |
| 总上下文窗口 | 8192 tokens |
| 单次 assistant 生成 | 最多 768 tokens |
| 单次 top-3 observation | 最多 768 tokens |
| 训练集 | 15,000 条唯一问题轨迹 |
| teacher-forced eval | 1,000 条，与 train 按 question ID 隔离 |
| 候补审计池 | 目标至少 2,000 条 |
| SFT | 从 post-trained 4B 独立开始的 Full SFT；不把 LoRA 作为必经步骤 |
| RL | R3.0 SFT 通过端到端准入门槛前不启动 |

机器可读单一真源为 `configs/protocols/searchqa_repro_v3_0_0.json`，当前 SHA-256 为 `69e21ed32b2a9864c7021914d4a3eac0e56a660acd0a094cce2483c390e43fd4`。代码、轨迹、数据 manifest、checkpoint 和评测 summary 都必须写入同一个 `protocol_id` 及配置 SHA-256；路径命名统一使用 `searchqa_repro_v3_0_0`。任何配置改动都必须同步更新版本和校验和。

### Canonical Prompt

Teacher 生成、SFT 和端到端评测都必须从 parquet 的 `prompt[0].content` 读取同一原文，不能在不同脚本中重新拼写。当前数据中的冻结文本为：

```text
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

代码块中的问题行末还包含一个固定 LF（`\n`），机器配置必须保留。为了复现 token 序列，第一版保留原文中的 `as your want`，不做文案润色。不增加额外 system prompt，也不在每次检索后追加 `Use the evidence above`、`final allowed search` 等普通提示。

### 轨迹序列化与工具返回

1. 使用对应 Qwen3.5 tokenizer/chat template 只渲染一次初始 user prompt，并加入 generation prefix。
2. 后续 `<think>/<search>`、环境注入的 `<information>`、下一轮 `<think>/<search>` 或 `<think>/<answer>` 依时间顺序追加到同一条连续序列；`<information>` 不包装成新的 user turn。
3. observation 统一序列化为：

```text
<information>
Doc 1(Title: {title}) {text}
Doc 2(Title: {title}) {text}
Doc 3(Title: {title}) {text}
</information>
```

4. 原始候选记录必须额外保存 doc ID、rank、BM25 score、未截断文本和实际输入文本。模型可见文本不要求暴露内部 doc ID。
5. 上下文累计保留所有历史 query、结构标签和实际送入模型的 evidence。不得再使用“问题 + 最近一次交换”的策略。

### Token 预算与截断语义

Qwen3.5-4B/27B 本地配置支持 262,144 token，但 v3 的操作窗口固定为 8192，以控制 vLLM KV cache、SFT activation 成本和吞吐。8192 不是模型能力上限，而是本实验的资源/协议上限。

预算按用途拆分，不能把一个 `max_tokens` 同时解释为全部限制：

| 预算 | 固定值 | 语义 |
|---|---:|---|
| `max_model_len` / SFT `cutoff_len` | 8192 | prompt + 历史 action + 历史 observation + 当前生成的总上限 |
| 初始 prompt 设计预算 | 512 | chat template、工具说明、问题；超过即数据 QA 失败，不通过左截断修复 |
| `max_new_tokens_per_action` | 768 | 每次 `<think>...<search/answer>` 的输出硬上限 |
| `max_observation_tokens` | 768 | 一次 top-3 `<information>` 整块的硬上限 |
| `max_search_actions` | 4 | 合法、重复或无进展的 search action 都消耗预算 |
| answer-only generation | 1 | 第 4 次搜索后只允许回答，不增加搜索额度 |

最坏设计包络约为 `512 + 4 × (768 action + 768 observation) + 768 final answer = 7424` tokens，低于 8192，并保留约 768 token 的实现/模板余量。正常样本不应触发总上下文裁剪。

observation 必须按实际 tokenizer 做 token 级裁剪，而不是按 Python 字符数裁剪。标题、rank 和三篇文档的存在优先保留；正文预算在三篇文档间均衡分配，短文档未使用的预算可以确定性地让给长文档，最终断言整个 `<information>` 不超过 768 token。不能简单保留拼接块的前 768 token，导致 Doc 3 系统性消失。裁剪算法不得使用 gold/supporting facts 做 oracle 选句；supporting-document 和 supporting-sentence 可见性必须在裁剪后的实际模型输入上审计。

assistant 输出遇到 `</search>` 或 `</answer>` 立即停止。如果达到 768 token 仍没有闭合合法动作，标记 `generation_truncated`：Teacher 候选不得进入 SFT，端到端评测记为失败；禁止补标签或拼接成伪完整轨迹。

SFT 构建阶段对每条 canonical 序列使用 Student tokenizer 精确计数。超过 8192 token 的候选必须留在原始候选池并以 `sequence_overflow` 拒绝，不能静默截断问题、早期证据、目标 action 或最终答案。评测阶段若实现断言发现输入加预留输出将超过 8192，应记录 `context_overflow` 并失败；pilot 验收要求该比例接近 0，再决定是否需要建立新的长上下文协议。

`cutoff_len=8192` 只表示允许的上限，不表示把每条训练样本预填充到 8192。训练应动态 padding 到当前 batch 的最长样本，并尽量按长度分桶，以免短轨迹为未使用 token 支付显存和计算成本。若启用 sample packing，必须先验证轨迹边界、attention mask 和 information loss mask 不会跨样本污染；第一轮 smoke test 默认不依赖 packing。

#### Token 预算的现有数据依据

2026-09-16 使用 `/data0/xs/models/Qwen3.5-4B` tokenizer，对旧合并 Teacher 轨迹每 7 条抽 1 条，共 2,637 条进行诊断；该样本只用于定预算，不用于声称新数据分布：

| 部分 | mean | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| 初始 canonical prompt | 147.8 | 169 | 184 | 210 | 273 |
| 单个 information block | 470.0 | 517 | 533 | 573 | 658 |
| 两个 information 之间的 assistant segment | 203.0 | 379 | 491 | 672 | 767 |
| prompt + 完整旧轨迹 | 1290.5 | 1995 | 2329 | 3420 | 5353 |

因此 Search-R1 上游的 500-token action/observation 默认值会截断一部分现有真实输出，而 768 基本覆盖观察到的 p99；新数据又将显著提高 2+/3+ 搜索比例，继续使用 4096 总窗口会增加早期多跳证据被裁剪的风险。8192 是本项目当前更稳妥的主配置。若未来需要与原始上游做严格 token-budget 消融，可另建 `v3_4k_compat`，不能混入主结果。

### 搜索、重复与计数

1. 每个合法 `<search>` 都增加 `search_action_count`，包括重复 query 和返回旧文档的 query；这是 v3 的主要“平均检索次数”口径。
2. 相同 query 不终止，可以从确定性缓存返回相同 top-3，但仍计入 action/cost；不向模型提供带有答案信息的纠错提示。
3. 不同 query 返回相同文档也继续运行并计为 no-progress。
4. 同时报告 `bm25_execution_count`、`unique_query_count`、`retrieved_document_count`、`unique_document_count`、`progressive_search_count` 和 `no_progress_search_count`。
5. 达到第 4 次搜索后保留一次 answer-only generation；若再次输出搜索，不执行并以 `search_limit` 结束。

### Teacher 候选生成与保存

pilot 使用冻结 manifest `data/processed/searchqa_repro_v3_0_0/teacher_pilot_manifest_2k_seed42.jsonl`：2,000 个归一化后唯一问题，HotpotQA/NQ=1,400/600，HotpotQA 内按原始 bridge/comparison × easy/medium/hard 比例分层；它排除固定 5k interactive dev 以及固定 50k final test 的所有同题 hash。manifest SHA-256 为 `c6ad15a0c96b356e0f03fe30fa9da28963db58a682e747c8d60d13f20e979d71`。每题用不同确定性 seed 采样 4 条候选，形成 8,000 个 rollout。初始解码为 `temperature=0.7`、`top_p=0.9`；所有候选都落盘，不能只保存通过项。

候选记录至少保存：稳定 question ID、source、split、gold aliases、Hotpot type/level/supporting facts、原始输出、canonical 序列、逐轮 prompt token 数、query、doc ID/rank/score、截断前后 evidence、最终答案、终止原因、所有过滤标签、模型/Prompt/protocol/tokenizer/retriever 版本和 seed。

pilot 先测严格合格率，再按 `目标 18,000 / 严格合格率 × 1.2` 估计扩采规模；后续按缺口定向补采，不能继续按生成顺序取前 15,000 条。

### 硬拒绝规则

以下任一条件成立时，轨迹不进入正式 SFT：

- 标签缺失、嵌套或顺序错误，动作外存在不允许文本，模型伪造 `<information>`；
- 空 search、空 answer、超过 4 次搜索、生成截断、上下文溢出或检索异常；
- 最终答案按非空 normalized EM 与所有 gold/alias 都不一致；
- 没有任何检索，只有闭卷正确答案；
- query 完全重复，或不同 query 没有带来任何新文档/新证据；
- query 与记录的 information 不对应；
- question ID 重复或与 teacher-forced eval、interactive dev、final test 泄漏。

生成环境允许含重复/无进展动作的候选继续运行，以便分析恢复能力；但正式 SFT 数据仍严格排除这些轨迹。不得从原轨迹中手术式删除坏步骤后冒充 Teacher 原始轨迹；应选择同题的另一条干净 rollout，或重新采样。

### 证据质量与候选排序

HotpotQA 使用数据集直接提供的两篇 supporting-document 标题和 supporting sentence ID：

- A 级：答案正确、行为干净、两篇 gold supporting documents 都进入模型可见上下文；在语料能够对齐 supporting sentence 时，还需确认关键 supporting sentence 未被 observation 截断；后续搜索产生新证据；
- B 级：答案和行为正确，但只覆盖一篇 gold supporting document，另一跳可能依赖参数知识或非 gold 文档。

正式选集中的 HotpotQA 轨迹至少 85% 为 A 级、B 级最多 15%，C 级不进入；若 A 级不足则继续扩采，不能静默放宽。一次搜索若在 top-3 中同时覆盖两篇 gold 文档，属于真实高效轨迹，不能为增加轮数而删除。NQ 不强制多轮，但至少要求一次有效检索；最终 NQ 中至少 90% 要能在裁剪后的可见 evidence 中找到 normalized gold answer/alias，其余未直接字符串对齐的轨迹单列审计。

同一问题的多条硬性合格轨迹按以下词典序选择：完整证据覆盖、每轮有新证据、无截断、在同等覆盖下更少的无效成本和 token。不能用“更短”压过证据更完整的轨迹。

最终候选池目标不少于 18,000 条；确定性、按 question ID 分层选择 15,000 train + 1,000 teacher-forced eval，保留至少 2,000 条候补。组成冻结为 HotpotQA/NQ 约 70%/30%，搜索桶目标为 1 次约 40%、2 次约 35%、3–4 次约 25%，并与上述证据等级做交叉分层；HotpotQA 内还要尽量保持原始 bridge/comparison 和 easy/medium/hard 分布。所有较长轨迹必须有证据新增，不得以冗余查询人工填长。

### SFT 表示与 Loss

每条轨迹作为一条与运行时一致的连续序列训练，不再把每一步改造成不同角色语义的伪 user turn。使用自定义预处理/collator 生成精确 span mask：

| token span | loss |
|---|---:|
| system/user/chat-template prompt | mask (`-100`) |
| 模型生成的 `<think>...</think>` | 计算 |
| 模型生成的 `<search>...</search>` | 计算 |
| 环境注入的 `<information>...</information>` | mask (`-100`) |
| 模型生成的 `<answer>...</answer>` | 计算 |
| padding | mask (`-100`) |

LLaMAFactory 的 `train_on_prompt=false` 不能单独证明 assistant 内部 information 已正确 mask；正式训练前必须对 tokenizer 后的 `input_ids/labels` 做自动断言和可读抽查。

Qwen3.5 官方 chat template 会在第一轮 assistant generation prefix 末尾自动加入 `<think>\n`。R3.0 保存的 canonical 首轮事件仍保留完整 `<think>...</think>`，但 runtime/SFT token 序列复用模板提供的第一个 opener，只监督首轮 think 正文和闭合标签；后续轮的完整 `<think>` 标签均由模型生成并参与 loss。实现必须断言不会产生 `<think>\n<think>` 双 opener。

### 独立 interactive dev

固定 50k final test 几乎占满 test parquet，剩余样本不足以构造 5k dev。因此 R3.0 在 Teacher 采样前从 train parquet 固定保留 5,000 题，并从所有 Teacher/SFT 候选中排除：

- 完整 dev：`data/processed/search_eval/searchqa_v3_interactive_dev_5k_seed43_train_holdout.jsonl`，HotpotQA 3,500 + NQ 1,500，SHA-256（canonical rows）`469ab57e52376abae3f7f17c4c9e016e1dbc8fb68eebf5a7e3bc1e61e4308341`；
- 快速 dev：上述固定集合的前 1,000 题，HotpotQA 707 + NQ 293，SHA-256（canonical rows）`0e5e5586f248e10a0f10184f19ff4d1b9a6cb8a6528480dc80cdf284fdc714f4`。

每 500 optimizer steps 暂停训练并释放 GPU，用四卡评测 quick dev，结果上传 SwanLab 后从同一 checkpoint 恢复；epoch 末/最终候选在完整 5k dev 上评测。这样不会在 ZeRO-3 训练仍占显存时并发启动 vLLM，也不会改变完整训练的 scheduler 总步数。

Full SFT 的 cosine scheduler 从一开始按最多 2 epochs/3,750 steps 构建，但编排器在第 1 epoch 的 1,875 step 强制保存并停止。第 2 epoch 不是默认动作：只有 interactive dev 与 teacher-forced loss gate 通过后，才显式批准并从可恢复的 `checkpoint-1875` 延续同一 scheduler；禁止先按 1 epoch 将学习率退火到零、再临时改总步数续训。

### 数据与评测报告

候选漏斗必须报告总候选、格式、答案、工具有效性、重复/no-progress、Hotpot 双支持覆盖、NQ evidence grounding、1/2/3/4 次搜索、每轮新文档、截断/溢出和各拒绝原因；拒绝原因允许多标签。

最终数据必须报告 source/type/level、搜索轮数、search actions、unique queries/docs、Hotpot support recall、NQ answer-evidence coverage、token 长度 p50/p90/p95/p99/max、有效监督 token 比例和 information mask 比例。

最终评测的主指标为非空 EM、token F1、`search_action_count`、正确样本平均搜索 action、Hotpot support recall、格式合规和 answer rate；BM25 实际执行、unique query/doc、重复/no-progress、预算耗尽和停止原因作为辅助指标。未训练的 post-trained 4B 和新 Full SFT 必须在完全相同的 v3 协议下重测；v0/v1 数字只作历史参考。
