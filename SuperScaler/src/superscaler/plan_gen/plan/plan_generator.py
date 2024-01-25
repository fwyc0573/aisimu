# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Author: v-cluo
Version: 0.1
Update date: 07/22/2019

Main model of Plan Generator project.
This module will parsed nodelist, then generate the execution plan

v0.1 supported function:
Introduce import package.
'''

from superscaler.plan_gen.plan.plan_mapper import GPURoundRobinMapper
from superscaler.plan_gen.plan.plan_pool import PlanPool
from superscaler.plan_gen.plan.ring_allreduce_plan import RingAllreducePlan
from superscaler.plan_gen.plan.raw_allreduce_plan import RawAllreducePlan
from superscaler.plan_gen.plan.reduce_broadcast_allreduce_plan import \
     ReduceBroadcastAllreducePlan
from superscaler.plan_gen.plan.plan_manager import PlanManager


class PlanGenerator():
    def __init__(self, nodelist, resource_pool):
        ''' Init PlanGenerator with nodelist and a resource_pool

        Args:
            nodelist: a list node parsed from machine learning platform
            resource_pool: a ResourcePool class containing device info and
                router info
        '''
        self.__resource_pool = resource_pool
        self.__nodelist = nodelist

        # RingAllreducePlan用于挑选在parser中挑选出的allreduce节点
        # TODO: 除了ring以外其他plan_name是什么作用？
        self.__plan_pool = PlanPool()
        self.__plan_pool.add_plan(RingAllreducePlan(plan_name='ring'))
        self.__plan_pool.add_plan(RawAllreducePlan(plan_name='raw'))
        self.__plan_pool.add_plan(
            ReduceBroadcastAllreducePlan(plan_name='ReduceBroadcast'))
        
        # mapping planned device to real GPU（当前DDP只涉及一个device，对应唯一一个GPU）
        self.__mapper = GPURoundRobinMapper(resource_pool)
        # Init PlanManager by PlanPool and PlanMapper
        self.__planmanager = PlanManager(self.__plan_pool, self.__mapper)

    def get_execution_plan(self, plan_type, plan_name):
        ''' Generate the execution plan by plan_type and plan_name
        Args:
            plan_type: string, e.g. Default
            plan_name: string, e.g. Default_Plan
        '''
        execution_plan =\
            self.__planmanager.get_execution_plan(self.__nodelist,
                                                  plan_type,
                                                  plan_name)

        return execution_plan

    def get_links_info(self):
        # Generate Link info
        links_info = self.__resource_pool.get_links_as_list()

        return links_info

    def get_routing_info(self):
        # Generate all routing info
        # Note the get_routing_info() function should be called after
        # the get_execution_plan() function, otherwise the routing_info
        # is empty
        routing_info = self.__mapper.route_info

        return routing_info

    def get_device_info(self):
        # Generate devices info
        device_info = self.__resource_pool.get_computational_hardware_as_list()

        return device_info

