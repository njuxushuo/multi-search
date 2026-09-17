# Multi-turn Search QA Reproduction

复现 Qwen3.5-27B Teacher → Qwen3.5-4B Student 的 Wikipedia 2018 + BM25 多轮检索问答、SFT 冷启动与后续成本感知 RL 项目。当前结论是继续改进 Teacher 数据与 Full SFT，暂不启动 RL。

## 文档导航

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)：当前真实状态、已完成内容、已知问题和产物路径。
- [`docs/ROADMAP.md`](docs/ROADMAP.md)：接下来的唯一执行计划和验收门槛。
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)：历史 v0/v1/v2 口径与已冻结的 R3.0 完整复现实验协议。
- [`docs/RESULTS.md`](docs/RESULTS.md)：已完成实验的正式结果，不记录运行中指标。

当前放行的下一轮协议为 **R3.10**（`searchqa_repro_v3_10_0`）：64×4 Teacher smoke 已通过全部质量门槛，正在进入 2,000×4 pilot。R3.0–R3.9 的配置和输出独立保留，只作诊断消融，不混入正式候选池。

## 环境入口

- Qwen3.5 生成/评测：`search-r1` Conda 环境，vLLM 0.18.0，使用 Triton GDN backend。
- SFT：LLaMA-Factory 本地源码与仓库内 `configs/sft_qwen35_4b_*.yaml`。
- 检索服务：`scripts/run_bm25_server.sh`。

环境细节、已知兼容性问题和可复现路径见 `docs/PROJECT_STATUS.md`。
