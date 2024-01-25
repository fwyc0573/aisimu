
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import sys
from collections import defaultdict
from functools import reduce
from statistics import mean
# from superscaler.plan_gen import ResourcePool, PlanGenerator, \
#     AISimulatorAdapter
# from superscaler.ai_simulator import Simulator

from superscaler.plan_gen import ResourcePool, PlanGenerator, \
    AISimulatorAdapter
from superscaler.ai_simulator import Simulator


class BenchmarkTools():

    def __init__(self, models, model_zoo, skip_coverage, skip_accuracy, config):
        self.__models = models
        self.__model_zoo = model_zoo
        self.__baseline = model_zoo.get_baseline()
        self.__skip_accuracy = skip_accuracy
        self.__skip_coverage = skip_coverage
        self.__coverages = defaultdict(list)
        self.__sim_time = defaultdict(dict)
        self.__baseline_time = defaultdict(dict)
        self.__percent = defaultdict(dict)
        self.__percent_model = defaultdict(list)
        self.enviroments = config['enviroments']
        self.resource_pool_path = config['resource_pool_path']

    def run(self):
        ''' Run BenchmarkTools
        '''
        if not self.__skip_accuracy:
            self.check_accuracy()
        if not self.__skip_coverage:
            self.check_coverage()

    def init_simulator(self, node_list, rp):
        '''plan_generator

        self.__mapper = GPURoundRobinMapper(resource_pool)
        self.__planmanager = PlanManager(self.__plan_pool, self.__mapper)
        确定plan devices 和 real GPU的映射方式（例如轮询，目前DDP模式只有一个device只mapping一个GPU）
        确定plan的类型（例如ring，raw，reduce_broadcast）
        '''
        plan_generator = PlanGenerator(node_list, rp)
        mapped_plan = plan_generator.get_execution_plan('Allreduce', 'ring').to_json()  # noqa: E501
        links_info = plan_generator.get_links_info()
        routing_info = plan_generator.get_routing_info() # 暂时都在/GPU/0里头，所以该dict为empty
        compu_device_spec = plan_generator.get_device_info()

        adapter = AISimulatorAdapter()
        assert adapter.set_plan(mapped_plan) is True
        sim_nodes_list = adapter.get_plan()

        # packages computing devices and network info into device_list
        device_list = [
            ('NetworkSimulator', ['NetworkSimulator', links_info, routing_info])
        ]
        for device_spec in compu_device_spec:
            device_list.append(
                (device_spec['type'],
                [device_spec['name'], device_spec['performance']])
            )

        sim = Simulator(sim_nodes_list, device_list)

        return sim

    def init_resource_pool(self, rp_path):
        '''get an example resource pool
        '''
        resource_yaml_path = os.path.join(rp_path)
        rp = ResourcePool()
        rp.init_from_yaml(resource_yaml_path)
        return rp

    def get_baseline_time(self, gpu, model):
        if str(gpu) in self.__baseline:
            if model in self.__baseline[str(gpu)]:
                baseline_time = self.__baseline[str(gpu)][model]
        else:
            baseline_time = 1.0
        return baseline_time

    def get_graph_path(self, model, gpu):
        if gpu == 1:
            graph_dir = self.__model_zoo.get_graph_path(model)
        else:
            graph_dir = self.__model_zoo.get_graph_path_multi_gpu(model)

        return graph_dir

    def get_parser(self, model):
        db_path = self.__model_zoo.get_database_path(model)
        nccl_dataset = self.__model_zoo.get_nccl_dataset()
        parser = self.__model_zoo.parser(db_file_path=db_path, nccl_dataset=nccl_dataset)

        return parser

    def get_node_list(self, model, gpu):
        graph_path = self.get_graph_path(model, gpu) # ai_simulator/simulator_benchmark/data/torch/graphs/bert_large_b16.json
        parser = self.get_parser(model) # 1
        # TODO： 没有区分device是GPU还是CPU，皆默认是device_0（压根没有传入device参数）
        nodelist = parser.parse_graph(graph_path, gpu=gpu) # def parse_graph(self, graph, device="device_0", gpu = 1)

        # TODO： 以下目前没作用
        new_nodelist = []
        for device_num in range(1):
            for i in nodelist:
                item = i.copy()
                # TODO: "network" 用来做什么的？
                if "network" in item['device']:
                    item['device'] = 'device_' + str(device_num) + "_network"
                else:
                    item['device'] = 'device_' + str(device_num)
                new_nodelist.append(item)
        # judge = self.compare_nodelists(nodelist, new_nodelist)
        # print(f"compare_nodelists = {judge}")
        # raise 0
        return new_nodelist


    def compare_nodelists(self, nodelist, new_nodelist):
        if len(nodelist) != len(new_nodelist):
            return False

        for i in range(len(nodelist)):
            if nodelist[i] != new_nodelist[i]:
                return False

        return True
    

    def check_model_accuracy(self, model, gpu):
        '''return the total time use of a model in simulator
        '''
        # TODO: Node's attrs['execution_time'] 是用来干什么的?

        nodelist = self.get_node_list(model, gpu)
        if nodelist is None:
            return None
        
        # 很关键的2步骤初始化资源池（读取配置），再进行模拟器的初始化
        rp = self.init_resource_pool(self.resource_pool_path)
        simulator = self.init_simulator(nodelist, rp)
        # TODO: 为什么使用time_use作为最终时间？最后一次完成的node即为总用时的意思？
        time_use, start_time, finish_time = simulator.run()

        return time_use

    def check_accuracy(self):
        '''print out the accuracy data of models
        '''
        """ Print head info for accuracy test
        """
        print('\n\nTimeuse(ms) and Accuracy(%) of simulator:')
        print('%-20s' % 'Environment' +\
              reduce(lambda x, y: x + y, (map(lambda x: '%-20s' % x, self.__models))))

        self.generate_accuracy()

        """ Print summary info for accuracy test
        """
        # print('%-20s' % 'average_loss', end='')
        # for model in self.__models + ['average']:
        #     print('%-20s' % ('%.1f%%' % mean(self.__percent_model[model])), end='')
        print()
        # print('Total average absolute loss: %.1f%%' % mean(self.__percent_model['average']))
        print('Mean Percentage Error (MPE): %.1f%%' % mean(self.__percent_model['MAP']))
        print('Mean Absolute Percentage Error (MAPE): %.1f%%' % mean(self.__percent_model['average']))
        sys.stdout.flush()

    def generate_accuracy(self):
        """ Generate the node converage and op converage
        """
        for env, gpu in self.enviroments.items():
            for model in self.__models:
                
                '''
                    这里主要得到2个value:__sim_time[env][model]和__baseline_time[env][model]
                    再根据这2个属性计算一系列其它属性
                '''

                # sim_time调用了simulator的run()函数
                sim_time = self.check_model_accuracy(model, gpu) / 1000
                if sim_time is not None:
                    self.__sim_time[env][model] = sim_time
                else:
                    self.__sim_time[env][model] = 1.0


                # TODO: 现有变量里头已记录了每个model的baseline_time
                self.__baseline_time[env][model] = self.get_baseline_time(gpu, model)

                # TODO: baseline在这里到底指的是什么？是实验对比的baseline，还是论文中的基础运行时间？
                self.__percent[env][model] =\
                    self.__sim_time[env][model] / self.__baseline_time[env][model] * 100
                self.__percent_model[model].append(abs(100-self.__percent[env][model]))
                self.__percent_model[env].append(abs(100-self.__percent[env][model]))
                # 按照这儿理解，一个是real word的record，一个是simulator的record。
                self.__percent_model['MAP'].append(100-self.__percent[env][model])
            self.__percent_model['average'].append(mean(self.__percent_model[env]))
            self.__percent_model['MAP'].append(mean(self.__percent_model[env]))

            print('%-20s' % env, end='')
            for model in self.__models:
                accuracy = '%.1f' %\
                    (self.__sim_time[env][model])
                print('%-20s' % accuracy, end='')
            print()
            print('%-20s' % env, end='')
            for model in self.__models:
                accuracy = '%.1f' %\
                    (self.__baseline_time[env][model])
                print('%-20s' % accuracy, end='')
            print()
            print('%-20s' % env, end='')
            for model in self.__models:
                accuracy = '%.1f%%' %\
                    (self.__percent[env][model])
                print('%-20s' % accuracy, end='')

            print()
            # print('%.1f%%' % mean(self.__percent_model[env]))

    def check_model_coverage(self, model, env, gpu):
        '''check if the 'execution_time' arg of nodes from graph
        exist in database
        '''
        nodelist = self.get_node_list(model, gpu)
        if nodelist is None:
            print('%-20s%-11s%-20s%-20s%s' % (env, model, 'xx', 'xx', ''))
            return None
        # TODO：exe_compu_nodes和total_compu_nodes的区别
        exe_compu_nodes = 0
        total_compu_nodes = 0
        op_exist = set()
        non_exist_ops = defaultdict(int)
        commu_nodes = [
            'Send', 'Recv', 'Allreduce'
        ]

        # count the nodes and ops with or without execution_time
        for node in nodelist:
            # 跳过通信节点，记录计算节点
            if node['op'] not in commu_nodes:
                total_compu_nodes += 1
                op_exist.add(node['op'])
                # 若存在execution_time属性
                if 'execution_time' in node.keys():
                    exe_compu_nodes += 1
                else:
                    # 如果不存在，则记录操作类型和频次；
                    # TODO:为什么会有operation没有execution_time属性？
                    non_exist_ops[node['op']] += 1

        total_compu_ops = len(op_exist)
        exe_compu_ops = len(op_exist) - len(non_exist_ops)

        # node_coverage是指有执行时间的计算节点与总计算节点的比例
        node_coverage = '%d/%d=%.1f%%' % (exe_compu_nodes,
                                        total_compu_nodes,
                                        exe_compu_nodes/total_compu_nodes*100)
        # op_coverage是指数据库中有记录的操作类型与总操作类型的比例。
        op_coverage = '%d/%d=%.1f%%' % (exe_compu_ops,
                                        total_compu_ops,
                                        exe_compu_ops/total_compu_ops*100)
        
        non_exist = ''
        for n, op in enumerate(non_exist_ops.items()):
            if (n >= 3):
                non_exist += '...'
                break
            non_exist += '%s(%d) ' % (op[0], op[1])
        print('%-20s%-11s%-20s%-20s%s' % (env, model, node_coverage, op_coverage, non_exist))

        # 返回的是node_coverage和op_coverage
        return (exe_compu_nodes/total_compu_nodes, exe_compu_ops/total_compu_ops)

    def check_coverage(self):
        '''check if the 'execution_time' arg of nodes from graph
        exist in database：评估模型中各种操作nodes的模拟执行时间是否由数据库条目覆盖。
        '''
        """ Print head info for coverage test
        """
        print('\n\nCoverage of database')
        print('%-20s%-11s%-20s%-20s%s' % ('environment','model', 'node coverage', 'OP coverage',
                                          'uncoveraged OP(count)'))

        # 更新self.__coverages['node']和self.__coverages['op']
        self.generate_coverages()

        """ Print summary info for coverage test
        """
        if 'node' in self.__coverages:
            print('Average node coverage: %.1f%%' %\
                (mean(self.__coverages['node']) * 100))
        if 'op' in self.__coverages:
            print('Average op coverage: %.1f%%' %\
                (mean(self.__coverages['op']) * 100))

    def generate_coverages(self):
        """ Generate the node converage and op converage
        """
        self.__coverages = defaultdict(list)
        for env, gpu in self.enviroments.items():
            for model in self.__models:
                # 返回（node_coverage，op_coverage）
                result = self.check_model_coverage(model, env, gpu)
                if result is not None:
                    self.__coverages['node'].append(result[0])
                    self.__coverages['op'].append(result[1])

