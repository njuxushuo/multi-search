# 初始化状态

- Search-R1 参考代码：`third_party/Search-R1`
- 检索入口：`third_party/Search-R1/example/retriever/retrieval_launch_bm25.sh`
- 官方下载脚本：`third_party/Search-R1/scripts/download.py`（下载 HF 上的 wiki-18 corpus 与 E5 索引）
- 本地目录：`data/raw`、`data/processed`、`data/corpus`、`data/index`

## SFT 框架选择

Search-R1 自带 `verl/trainer/fsdp_sft_trainer.py`，使用它最利于保持后续 RL 数据格式一致。LLaMA-Factory 也可以用于 LoRA/QLoRA SFT，但需要把多轮 tool-use 轨迹转换成其对话格式，并在进入 veRL 前重新转换为 parquet；建议第一版使用仓库自带 SFT，第二版再用 LLaMA-Factory 做对照。

## 版本注意

仓库原始 requirements 针对较旧的 `transformers<4.48` 和 `vllm<=0.6.3`。当前环境使用更新版本，启动训练前应先做一次兼容性 smoke test；不要直接降级，以免破坏已配置的 CUDA/PyTorch 环境。
