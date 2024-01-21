# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import yaml
from model_zoo import ModelZoo
from benchmark_tools import BenchmarkTools

# set up the parser
parser = argparse.ArgumentParser(
    prog='python3 simulator_benchmark.py',
    description='Run AI Simulator with 8 models benchmark',
    )

parser.add_argument('--skip-coverage',
                    dest='skip_coverage', action='store_true',
                    help='skip testing the databse coverage')

parser.add_argument('--skip-accuracy',
                    dest='skip_accuracy', action='store_true',
                    help='skip testting the simulator accuracy')

parser.add_argument('-c', '--config_path',
                    dest='config', default='config/config.yaml',
                    help='config setting.')


if __name__ == '__main__':
    args = parser.parse_args()

    # config包含了enviroments decive platform baseline_path resource_pool_path nccl_path task等信息
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_zoo = ModelZoo(config)


    models = model_zoo.get_model_list()

    benchmarktools = BenchmarkTools(models,
                                    model_zoo,
                                    args.skip_coverage,
                                    args.skip_accuracy,
                                    config)
    benchmarktools.run()


    # 30/10/2023 记录：
    # 近期主要解决问题：
    # 1. 梳理完架构总体流程逻辑，节点池是如何被一起控制的？
    # 2. 原始的测量数据是如何获取到的，在架构运行过程中哪些数据是被需要的，对应项目中哪些脚本，以及我该如何测试？
    
    '''
        10.30 2023

        BenchmarkTools.run()中check_coverage()函数没有特别复杂。需要关注的是check_accuracy()函数
        check_accuracy()主要涉及了Simulator的使用

        ->  Simulator学习

    
    '''


    '''
        10.31.2023

        Q：Training Script + Configurations => Graph Constructor的过程在哪？
        A：benchmark_tools的plan_generator = PlanGenerator(node_list, rp)

        Q:AISimulatorAdapter如何执行预测和干扰的？
        
        看AISimulatorAdapter()是怎么写的，以及原来的数据是如何测试的
    
    '''


    '''
        11.01.2023

        Q：MP场景该如何修改到PlanGenerator？这涉及到tensor和pipeline的两种情况，该如何交给设备？原始情况只有node->device？

        Q：对于现阶段混合模型，其MP部分的划分是由谁决定的，如何获取它的划分情况？是否有可能让Merak来优化模型的分割情况？数据并
           行确实没什么可挖的，但是模型并行，如何并行是最合理的？模拟出来以后再给出优化方案？

    '''


    '''
        11.08.2023
        Q：从哪个函数可以拿到完整的DAG图？以及如何拿到？

        Q：在AISimulatorAdapter中出现了拓扑关系函数（__create_index_dependency），不过看着貌似是name->ID，结构依旧是node_list里头的？
           如果涉及了DAG结构的重新组织，为什么要在accuracy中进行，这个结构无法被coverage共享吗？


    '''

    '''
        11.16.2023
        Q：simulator的149行exec_node.get_execution_time() 这一个预期数值是如何提前测试的？

        Q：simulator的161行函数中，如何计算earliest_complete_time的？

    '''


    '''
        11.17.2023
        Q：benchmark_tools.py的169行函数中，get_baseline_time指的是真实场景中的总时间？还是论文中的基础运行时间？

        Q：这儿计算了误差，那么修正误差以及ML部分在哪呢？

    '''


    '''
        11.18.2023
        Q：check_coverage函数的逻辑，以及存储的数据是什么。
        A：node_coverage是指有执行时间的计算节点与总计算节点的比例；op_coverage是指数据库中有记录的操作类型与总操作类型的比例。总体来说是用于衡量数据库中记录数据和模型之间的匹配程度

        Q: UserManual的Change the input resource中提到的的数据分别对应什么内容，是如何获取测试的？

    '''

    '''
        11.20.2023
        Q：(1) db content => 如何获取这些数据，测试脚本是哪个？
           (2) resource_pool.yaml => 如何设定，模型的结构如何写入？如何体现混合并行的结构？
        
        Q：论文中ML的预测，以及Tall−reduce = α + β + log2(N ) × γ + δ × tensor size的计算在哪？

        Q：论文中每个实验对应脚本一个个找过去。模型：BERT_Large、GPT2、VGG19、ResNet152
    '''


    '''
        11.27.2023
        P：在容器中测试one_click_test中的几个函数，验证其是否可以正常运行和测试。


    '''


    '''
        11.29.2023 【近阶段计划】
        
        P：测试27日计划的容器测试（涉及真实GPU的性能测试和分析部分）
        
        P：继续调研和测试 DP+MP的架构（pytorch2.0，keras3.0）使用方式，以及搜寻已经编写过的类似模型（主要围绕4个模型）;能否获取到开源架构对graph的定制？

        p：阅读simulator相关的论文，找到新的切入点
    
    
    '''

    '''



    '''