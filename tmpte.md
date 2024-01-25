# 测试文件记录

## ddp_profile_backup.sh，ddp_profile.py，ddp_profile.sh，profile.sh
利用DDP.json，多次执行模型的前向和后向传播，记录时间。

## ddp_test.py torch_dataset_test.py torch_graph_test.py
皆利用TorchGraph库（包含在项目中），生成DDP.json。
ddp_test.py 使用DDPGraph类，针对分布式数据并行（DDP）环境，特别是它包含了--local_rank 参数。构造和分析分布式数据并行（DDP）模型的图形表示。
torch_dataset_test.py 使用TorchDatabase类，不涉及分布式处理。主要目的在于性能分析，即收集和存储模型训练过程中的各种性能指标。
torch_graph_test.py 使用TorchGraph类，不涉及分布式处理，构建和分析模型的图形表示。
TODO:
Q：三者貌似存的都是DDP.json文件？

## one_click_test_backup.py 
实验核心脚本：批量测试模型训练时间的脚本，使用容器化操作，即批量化根据model和ddp.json运行ddp_profile.py进行测试。

## one_click_test.py 
核心脚本，一系列自动化操作，包含nccl_test、baseline_test、TorchGraph_test、ddpgraph_test、op_test测试。

## profile_test.py event-1.json log1.csv
对分布式训练环境中的模型执行性能分析。
脚本使用PyTorch提供的性能分析工具来收集和记录模型训练步骤中的CUDA时间。分析结果将以CSV和JSON格式保存。

## setup.sh 
容器内的安装环境，包括安装PyTorch、TorchGraph、nccl、nccl-tests、nvidia-docker、docker、docker-compose、nvidia-docker-compose

## 比较single_profile_test.py (或test.py)和 profile_test.py 和 torch_profiler.py
profile_test.py是在分布式环境下进行的性能分析，single_profile_test.py是在单机环境下进行的性能分析(没有NCCL初始化没有DDP)。
test.py 主要是简单的模型性能分析，它直接创建模型和数据，然后使用性能分析器分析模型的后向传播。
torch_profiler.py这个脚本看着没多大用，有的功能前两个脚本基本都实现了？里头import了一个额外的性能分析的库torchprof。

## standalone_allreduce_test.py 
使用PyTorch的性能分析器 (torch.profiler) 来测量和记录在分布式环境中执行all_reduce操作的性能。