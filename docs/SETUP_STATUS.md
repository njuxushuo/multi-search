# 初始化状态

## SFT 运行环境（2026-09-14）

- SFT 启动脚本统一使用 `conda activate search-r1` 对应的 `search-r1` 环境（当前 PyTorch 2.10.0+cu128、Transformers 4.57.6）。
- LLaMA-Factory 使用本机源码 `/data0/xs/LLaMA-Factory/src` 通过 `PYTHONPATH` 加载；DeepSpeed 兼容包暂从 llamafactory 环境的 site-packages 复用。启动入口为 `scripts/run_sft_qwen35_4b_full.sh`。
- 直接使用旧 `llamafactory` 环境恢复 checkpoint 会触发 Transformers 的 torch.load 安全版本限制，因此不再作为 SFT 运行环境。

- Search-R1 参考代码：`third_party/Search-R1`
- 检索入口：`third_party/Search-R1/example/retriever/retrieval_launch_bm25.sh`
- 官方下载脚本：`third_party/Search-R1/scripts/download.py`（下载 HF 上的 wiki-18 corpus 与 E5 索引）
- 本地目录：`data/raw`、`data/processed`、`data/corpus`、`data/index`

## 环境检查（2026-09-12）

- 推荐使用 Conda 环境：`search-r1` 当前 PyTorch 为 CUDA 13.0，与机器驱动不兼容。
- 可用环境：`atp`（Python 3.11.15、PyTorch 2.7.0、CUDA 12.6、Transformers 4.52.3），GPU 可正常识别，适合后续下载和检索准备。
- Wikipedia 2018 语料已从 ModelScope 镜像 `zhuoran997/wiki-18-corpus` 下载到 `data/corpus/wiki-18.jsonl.gz`，文件大小 5,123,307,260 字节，SHA-256 为 `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`。
- 该镜像文件是 gzip 包裹的 tar，已提取出可供 BM25 使用的 `data/corpus/wiki-18.jsonl`；首条记录已通过 JSON 解析校验。Hugging Face 主站仍无法直接连接，但不影响本地语料使用。
- `nq_hotpotqa_train` 已从 ModelScope 镜像 `zhuangzhuang2023/nq_hotpotqa_train` 下载到 `data/processed/nq_hotpotqa_train/`，包含 `train.parquet`（355,663,891 字节）和 `test.parquet`（70,370,337 字节），文件类型校验通过。
- `search-r1` 已按 `qwen35` 环境切换为 PyTorch 2.10.0（CUDA 12.8）、Transformers 4.57.6、vLLM 0.18.0，并以 editable、无依赖方式安装本地 veRL 代码。CUDA smoke test 和本地 Qwen3.5-2B vLLM 推理均通过。
- Qwen3.5 在当前系统 CUDA toolkit 下使用 FlashInfer GDN 预填充会触发 PTX 编译错误；运行 vLLM 时应设置 `gdn_prefill_backend='triton'`（命令行为 `--gdn-prefill-backend triton`）。
- 本地 Qwen3.5-2B 已用 `search-r1` + vLLM 0.18.0 完成单卡生成 smoke test（CUDA 12.8，使用 Triton GDN backend）。Search-R1 自带的 `verl` v0.1 rollout 适配器仍硬性限制 vLLM 0.3.1/0.4.2/0.5.4/0.6.3，直接导入会拒绝 vLLM 0.18.0；因此 Qwen3.5 的独立 vLLM 推理已可用，旧版 Search-R1 RL rollout 还需升级/适配 veRL 后才能训练。
- E5 索引分片已完整下载并合并为 `data/index/e5_Flat.index`（64,559,075,373 字节）。

## 环境隔离与可行路径（2026-09-12）

当前存在两条互不兼容的运行时链：

1. 原始 Search-R1 的本地 veRL v0.1 在 `verl/third_party/vllm` 中内置了 vLLM 0.3.1、0.4.2、0.5.4、0.6.3 的适配器，并在 rollout 初始化时拒绝其他版本。它不能在 vLLM 0.18.0 环境中直接运行。
2. Qwen3.5 已在当前 `search-r1` 环境通过 vLLM 0.18.0 完成单卡生成测试；该版本还需要 `gdn_prefill_backend=triton` 来避开当前 CUDA toolkit 下的 FlashInfer GDN PTX 编译问题。

因此，两个隔离的 Conda 环境是可行且推荐的，但它们分别承担不同任务，不能把旧训练器放在一个环境、再让同一进程跨环境调用 vLLM。当前机器已有可复用的 `text2sql-eval-vllm063`（Python 3.9.5、PyTorch 2.4.0+cu121、vLLM 0.6.3、CUDA 可用），可复制为 `search-r1-legacy`，用于原始 Search-R1/Qwen2.5 的复现实验；保留现有 `search-r1` 用于 Qwen3.5 的 vLLM 推理和服务。

已创建 `search-r1-legacy` 克隆环境，并再次验证其 PyTorch 2.4.0+cu121、vLLM 0.6.3 与当前 GPU 驱动可用；Search-R1 旧版的辅助依赖和 editable 安装仍需在该环境中单独完成，不能复用现代环境的核心包。

如果目标是 Qwen3.5 的 GRPO/PPO 训练，不能只靠环境隔离解决。需要另建基于上游新版 veRL 的训练环境，并把 Search-R1 的数据预处理、检索奖励和训练入口迁移到新版 veRL API；上游当前文档支持 vLLM 0.18.0 及更高版本，并按推理引擎锁定匹配的 Transformers。直接给旧版 vendored veRL 打 vLLM 0.18 补丁风险很高，不作为首选。

把现代 vLLM 作为独立 HTTP 服务、让旧训练环境调用它，只适合冻结模型的批量生成、评估或教师轨迹生成；由于无法自动完成旧版 PPO/GRPO 所需的进程内权重同步，不适合作为原始在线 RL rollout 的直接替代。

## SFT 框架选择

Search-R1 自带 `verl/trainer/fsdp_sft_trainer.py`，使用它最利于保持后续 RL 数据格式一致。LLaMA-Factory 也可以用于 LoRA/QLoRA SFT，但需要把多轮 tool-use 轨迹转换成其对话格式，并在进入 veRL 前重新转换为 parquet；建议第一版使用仓库自带 SFT，第二版再用 LLaMA-Factory 做对照。

## 版本注意

仓库原始 requirements 针对较旧的 `transformers<4.48` 和 `vllm<=0.6.3`。当前环境使用更新版本，启动训练前应先做一次兼容性 smoke test；不要直接降级，以免破坏已配置的 CUDA/PyTorch 环境。

## 按照附件规格的当前核对（2026-09-12）

附件要求是 Qwen3.5-27B Teacher → Qwen3.5-4B Student、15,000 条经过工具调用校验的多轮轨迹，并在独立跨数据集评测集上报告 EM 和平均检索次数。按此口径，准备工作尚未全部完成：

- 已完成：Wikipedia 2018 语料、HotpotQA/NQ parquet、E5 索引、4B 学生模型；`search-r1` 中的 faiss 与 SwanLab 导入及云端上传测试。
- 已完成：Qwen3.5-27B 教师权重通过 ModelScope 下载到 `models/Qwen3.5-27B`；11 个 safetensors 分片与索引逐一核对，无 `.incomplete` 文件。
- 已完成：E5 encoder safetensors 权重已补齐（避免下载无关 ONNX 变体）。
- 已完成：使用机器上的 Java 21 构建与官方检索服务兼容的 Lucene BM25 索引，共 21,015,324 篇文档，0 个索引错误；单查询 top-k 健康检查通过。
- 未完成：真实 15,000 条教师多轮轨迹生成、格式/工具/答案过滤，以及独立 50k 评测结果（评测脚本已固定）。
- 已固定：`configs/sft_qwen35_4b_lora.yaml` 和 `scripts/run_sft_qwen35_4b_lora.sh` 只接受 15,000 条校验轨迹；`scripts/validate_teacher_traces.py` 会拒绝没有 `<search>/<information>` 的 oracle-only pilot 数据。此前的 2B pilot 已停止，不计入正式结果。
- 已固定教师生成入口：`scripts/generate_teacher_traces.py` 使用本地 Qwen3.5-27B 和 Lucene BM25 逐轮执行 `<search> → <information>`；启动脚本默认生成 150,000 条候选（可用 `TEACHER_COUNT` 覆盖），只把最终答案匹配 gold 的轨迹交给校验器。
- vLLM 运行验证：Qwen3.5-27B 在 TP4、`gdn_prefill_backend=triton` 下可加载并生成；当前 PyTorch/vLLM 组合的 torch.compile 路径会触发 `FakeTensorMode` 兼容错误，因此教师生成默认使用 eager 模式。已修复 BM25 索引无 stored raw 时按 docid 回读 JSONL 正文，并以 2 条样本验证 1 条严格答案通过。

训练环境采用隔离方案：LLaMA-Factory 的 `llamafactory` 环境负责 Qwen3.5-4B SFT（Transformers 5.8），`search-r1` 环境负责现代 vLLM/检索依赖；这样避免原始 veRL v0.1 对旧版 vLLM 的硬性版本约束。训练脚本固定 `CUDA_VISIBLE_DEVICES=3`，因为核对时只有 GPU 3 真正空闲。

教师生成和 SFT 启动脚本均在启动前读取 `nvidia-smi`，若目标 GPU 显存占用超过 1 GiB 会直接退出；教师脚本还要求 `TEACHER_TP_SIZE` 与 `TEACHER_GPUS` 数量一致，最多可显式配置 4 张当时确实空闲的卡。
