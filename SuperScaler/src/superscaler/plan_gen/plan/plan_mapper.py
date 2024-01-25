# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import copy
from superscaler.plan_gen.plan.node_list import NodeList
from superscaler.plan_gen.plan.resources.resource_pool import ResourcePool


class PlanMapper(abc.ABC):
    """ An mapper class maps nodes on actual devices
    """

    def __init__(self, resource_pool):

        if not isinstance(resource_pool, ResourcePool):
            raise ValueError(
                "Input resource_pool must be ResourcePool instance")
        self.__resource_pool = resource_pool
        self.__routing_info = {}

    @property
    def resource_pool(self):
        return self.__resource_pool

    @property
    def route_info(self):
        return self.__routing_info

    def _reset_route_info(self):
        ''' Reset route_info as empty dict
        '''
        self.__routing_info = {}

    def _update_route_info(self, src_name, dst_name, index, path):
        ''' Updating self._route_info with new route, where the key is
            (src_name, dst_name, index), and val is link info on path
        '''
        self.__routing_info[(src_name, dst_name, index)] = \
            [link.link_id for link in path]

    @abc.abstractmethod
    def map(self, plan):
        return None


class GPURoundRobinMapper(PlanMapper):
    """ Assign device as GPURoundRobin
    """

    def __init__(self, resource_pool):
        super().__init__(resource_pool)
        self.__gpus = self.resource_pool.get_resource_list_from_type("GPU")
        self.__cpus = self.resource_pool.get_resource_list_from_type("CPU")

    def map(self, node_list):
        if not isinstance(node_list, NodeList):
            return None
        else:
            mapped_node_list = node_list
            self._reset_route_info()
            if not self.__assign_device(mapped_node_list):
                return None
            else:
                return mapped_node_list

    def __assign_device(self, node_list):
        ''' This function assigns the virtual devices of node_list
            as the real devices of resource_pool
            检查资源池中是否有足够的GPU来满足node_list中的设备需求。
            然后,使用RoundRobin算法将节点分配到GPU上。
        '''
        # Record all devices of node_list
        devices = []

        for node in node_list:
            if node.device is not None and node.device not in devices:
                devices.append(node.device)

        # Check whether the node_list can be assigned into resource_pool
        if len(self.__gpus) < 1:
            # Resource Pool is empty
            raise RuntimeError("Resource Pool is empty")
        
        # devices是虚拟设备（计划设备）；在DDP中，每个记录的trace一般是一个GPU内发生的，因此原工程涉及的devices的len()全部为1
        # 即将计算任务分配到一个GPU上，符合DDP逻辑
        if len(self.__gpus) < len(devices):
            # GPU count in resource_pool can't meet the requirement
            raise RuntimeError("GPU count in resource_pool can't meet the requirement")

        # Assign devices by RoundRobin order
        for node in node_list:
            src_gpu, dst_gpu = None, None
    
            # Assign device
            if node.device is not None:
                src_gpu = self.__gpus[devices.index(node.device)]
                node.device = src_gpu.get_name()  # DDP的话，都在一个GPU上，例如'/server/hostname0/GPU/0'
            
            '''现阶段下文2个部分都没涉及，即task graph中node.target都是None。
            
                ∵ DDP中的single world处理的是一样的，不需要跨GPU。
            
            '''
            # Assign target
            if node.target is not None:
                dst_gpu = self.__gpus[devices.index(node.target)]
                node.target = dst_gpu.get_name()
                raise 0
            # Assign route
            if node.device is not None and node.target is not None:
                node.route_index = 0
                route_path = self.resource_pool.get_route_info(
                    src_gpu.get_name(), dst_gpu.get_name())
                # Check all routes exists for all communication nodes
                # No route found between src_gpu and dst_gpu
                if not route_path:
                    self._reset_route_info()
                    raise RuntimeError("No route found between src_gpu and dst_gpu")
                print(f"route_path -> {route_path}")
                route_path = route_path[0]
                node.route_type = route_path[1]

                self._update_route_info(
                    src_gpu.get_name(), dst_gpu.get_name(), node.route_index, route_path[0])
                raise 0
        return True
