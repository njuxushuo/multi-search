# Multi-turn Search QA RL Reproduction

复现 Wikipedia + BM25 多轮搜索问答、SFT 冷启动与 DAPO/GRPO 优化项目。

## 初始化

```bash
conda create -n search-r1 python=3.10
conda activate search-r1
pip install -r requirements.txt
git clone https://github.com/PeterGriffinJin/Search-R1.git third_party/Search-R1
```

详细路线见 `docs/ROADMAP.md`。
