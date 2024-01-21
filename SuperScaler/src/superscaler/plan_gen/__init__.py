# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.plan.plan_generator import PlanGenerator
from superscaler.plan_gen.plan.resources.resource_pool import ResourcePool
from superscaler.plan_gen.plan.adapter.ai_simulator_adapter import \
     AISimulatorAdapter


# init 文件的用法，将所有的类都放在这里，然后在外部直接import init对应的父文件夹名，
# 就可以直接使用所有的类了（这些类原本都是子文件夹下py脚本中的类，在init中重新提前整理）

__all__ = ['PlanGenerator', 'ResourcePool', 'TFParser', 'TorchParser',
           'AISimulatorAdapter']

try:
     from superscaler.plan_gen.plan.parser.tf_parser import TFParser
     __all__.append('TFParser')
except:
     pass

try:
     from superscaler.plan_gen.plan.parser.torch_parser import TorchParser
     __all__.append('TorchParser')
except:
     pass

