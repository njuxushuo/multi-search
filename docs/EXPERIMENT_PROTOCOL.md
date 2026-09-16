# 实验记录规范

## 文档职责

- 本文件保存已经冻结、可复现的实验协议及指标口径。
- 近期实验的执行顺序保存在 `docs/ROADMAP.md`。
- 完成后的正式数字写入 `docs/RESULTS.md`，不要把运行中的中间值当作最终结果。

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

## v0 冻结评测协议（2026-09-15）

### 协议身份与用途

项目文档中的 **v0** 指当前正在使用的旧终止规则评测。代码写入结果 metadata 的内部协议名为 `qwen35_search_r1_v4_dp_doc_novelty`。v0 用于保存 Base、LoRA、Full SFT、27B Teacher 的第一版可比基线；后续修正重复搜索处理时必须产生新协议版本，不能覆盖 v0 文件。

截至 2026-09-16，Base、LoRA 和 Full SFT 的 v0/v1 本地评测均已完成并严格合并。Teacher 已改在其他服务器评测，本机顺序启动脚本不再启动 Teacher。Qwen3.5-4B-Instruct 不属于 v0 四模型集合，计划直接使用修正后的 v1 协议评测。

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
| v1 | 不立即终止，不重复调用 BM25，返回中性 no-progress observation | 有；最多 8 次搜索尝试后保留一次最终回答生成 | 原问题 + 最近一次有效检索；纠错生成时临时附带当前重复动作和中性 observation | 隔离评估终止规则的影响，并加入 4B-Instruct |
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
