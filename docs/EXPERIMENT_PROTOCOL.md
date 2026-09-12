# 实验记录规范

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
