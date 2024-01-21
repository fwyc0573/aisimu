# Details

Date : 2023-10-08 21:09:13

Directory e:\\HUBOther\\ML_Sys_Merak\\SuperScaler\\src

Total : 132 files,  25701 codes, 1361 comments, 2036 blanks, all 29098 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [SuperScaler/src/backend/common_runtime/executor/Dockerfile](/SuperScaler/src/backend/common_runtime/executor/Dockerfile) | Docker | 17 | 0 | 4 | 21 |
| [SuperScaler/src/backend/common_runtime/executor/README.md](/SuperScaler/src/backend/common_runtime/executor/README.md) | Markdown | 121 | 0 | 67 | 188 |
| [SuperScaler/src/backend/common_runtime/executor/channel/channel.hpp](/SuperScaler/src/backend/common_runtime/executor/channel/channel.hpp) | C++ | 19 | 20 | 7 | 46 |
| [SuperScaler/src/backend/common_runtime/executor/channel/cpu_channel.hpp](/SuperScaler/src/backend/common_runtime/executor/channel/cpu_channel.hpp) | C++ | 57 | 18 | 7 | 82 |
| [SuperScaler/src/backend/common_runtime/executor/channel/fifo.hpp](/SuperScaler/src/backend/common_runtime/executor/channel/fifo.hpp) | C++ | 67 | 3 | 6 | 76 |
| [SuperScaler/src/backend/common_runtime/executor/config.hpp](/SuperScaler/src/backend/common_runtime/executor/config.hpp) | C++ | 4 | 2 | 3 | 9 |
| [SuperScaler/src/backend/common_runtime/executor/cpu_kernels.hpp](/SuperScaler/src/backend/common_runtime/executor/cpu_kernels.hpp) | C++ | 34 | 2 | 7 | 43 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/channel_manager.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/channel_manager.hpp) | C++ | 73 | 2 | 11 | 86 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel.cpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel.cpp) | C++ | 393 | 18 | 50 | 461 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel.hpp) | C++ | 135 | 84 | 36 | 255 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel_defs.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_channel_defs.hpp) | C++ | 14 | 2 | 4 | 20 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_ipc_internal.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/cuda_ipc_internal.hpp) | C++ | 112 | 2 | 14 | 128 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/handle_manager.cpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/handle_manager.cpp) | C++ | 42 | 2 | 6 | 50 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/handle_manager.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/handle_manager.hpp) | C++ | 40 | 19 | 9 | 68 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/shared_block.cpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/shared_block.cpp) | C++ | 46 | 2 | 11 | 59 |
| [SuperScaler/src/backend/common_runtime/executor/cuda_ipc/shared_block.hpp](/SuperScaler/src/backend/common_runtime/executor/cuda_ipc/shared_block.hpp) | C++ | 23 | 2 | 10 | 35 |
| [SuperScaler/src/backend/common_runtime/executor/exec_ctx.hpp](/SuperScaler/src/backend/common_runtime/executor/exec_ctx.hpp) | C++ | 14 | 2 | 4 | 20 |
| [SuperScaler/src/backend/common_runtime/executor/exec_info.cpp](/SuperScaler/src/backend/common_runtime/executor/exec_info.cpp) | C++ | 20 | 2 | 7 | 29 |
| [SuperScaler/src/backend/common_runtime/executor/exec_info.hpp](/SuperScaler/src/backend/common_runtime/executor/exec_info.hpp) | C++ | 16 | 2 | 8 | 26 |
| [SuperScaler/src/backend/common_runtime/executor/executor.hpp](/SuperScaler/src/backend/common_runtime/executor/executor.hpp) | C++ | 23 | 38 | 8 | 69 |
| [SuperScaler/src/backend/common_runtime/executor/executor_pub.hpp](/SuperScaler/src/backend/common_runtime/executor/executor_pub.hpp) | C++ | 7 | 2 | 3 | 12 |
| [SuperScaler/src/backend/common_runtime/executor/gpu_kernels.cu](/SuperScaler/src/backend/common_runtime/executor/gpu_kernels.cu) | CUDA C++ | 73 | 2 | 15 | 90 |
| [SuperScaler/src/backend/common_runtime/executor/gpu_kernels.hpp](/SuperScaler/src/backend/common_runtime/executor/gpu_kernels.hpp) | C++ | 22 | 2 | 7 | 31 |
| [SuperScaler/src/backend/common_runtime/executor/poll_executor.cpp](/SuperScaler/src/backend/common_runtime/executor/poll_executor.cpp) | C++ | 141 | 5 | 29 | 175 |
| [SuperScaler/src/backend/common_runtime/executor/poll_executor.hpp](/SuperScaler/src/backend/common_runtime/executor/poll_executor.hpp) | C++ | 60 | 41 | 21 | 122 |
| [SuperScaler/src/backend/common_runtime/executor/rdma_tunnel.hpp](/SuperScaler/src/backend/common_runtime/executor/rdma_tunnel.hpp) | C++ | 63 | 2 | 7 | 72 |
| [SuperScaler/src/backend/common_runtime/executor/recv_task.cpp](/SuperScaler/src/backend/common_runtime/executor/recv_task.cpp) | C++ | 18 | 2 | 3 | 23 |
| [SuperScaler/src/backend/common_runtime/executor/recv_task.hpp](/SuperScaler/src/backend/common_runtime/executor/recv_task.hpp) | C++ | 21 | 2 | 5 | 28 |
| [SuperScaler/src/backend/common_runtime/executor/reduction_task.hpp](/SuperScaler/src/backend/common_runtime/executor/reduction_task.hpp) | C++ | 33 | 3 | 9 | 45 |
| [SuperScaler/src/backend/common_runtime/executor/samples/allreduce_sample.cpp](/SuperScaler/src/backend/common_runtime/executor/samples/allreduce_sample.cpp) | C++ | 136 | 2 | 29 | 167 |
| [SuperScaler/src/backend/common_runtime/executor/samples/cuda_channel.cpp](/SuperScaler/src/backend/common_runtime/executor/samples/cuda_channel.cpp) | C++ | 94 | 17 | 10 | 121 |
| [SuperScaler/src/backend/common_runtime/executor/samples/cuda_task.cpp](/SuperScaler/src/backend/common_runtime/executor/samples/cuda_task.cpp) | C++ | 81 | 3 | 15 | 99 |
| [SuperScaler/src/backend/common_runtime/executor/samples/handle_manager.cpp](/SuperScaler/src/backend/common_runtime/executor/samples/handle_manager.cpp) | C++ | 85 | 20 | 22 | 127 |
| [SuperScaler/src/backend/common_runtime/executor/samples/ring_allreduce_sample.cpp](/SuperScaler/src/backend/common_runtime/executor/samples/ring_allreduce_sample.cpp) | C++ | 221 | 17 | 32 | 270 |
| [SuperScaler/src/backend/common_runtime/executor/scale_task.hpp](/SuperScaler/src/backend/common_runtime/executor/scale_task.hpp) | C++ | 33 | 6 | 10 | 49 |
| [SuperScaler/src/backend/common_runtime/executor/send_task.cpp](/SuperScaler/src/backend/common_runtime/executor/send_task.cpp) | C++ | 19 | 2 | 3 | 24 |
| [SuperScaler/src/backend/common_runtime/executor/send_task.hpp](/SuperScaler/src/backend/common_runtime/executor/send_task.hpp) | C++ | 21 | 2 | 5 | 28 |
| [SuperScaler/src/backend/common_runtime/executor/task.cpp](/SuperScaler/src/backend/common_runtime/executor/task.cpp) | C++ | 71 | 3 | 13 | 87 |
| [SuperScaler/src/backend/common_runtime/executor/task.hpp](/SuperScaler/src/backend/common_runtime/executor/task.hpp) | C++ | 37 | 28 | 16 | 81 |
| [SuperScaler/src/backend/common_runtime/executor/task_manager.cpp](/SuperScaler/src/backend/common_runtime/executor/task_manager.cpp) | C++ | 28 | 2 | 8 | 38 |
| [SuperScaler/src/backend/common_runtime/executor/task_manager.hpp](/SuperScaler/src/backend/common_runtime/executor/task_manager.hpp) | C++ | 42 | 22 | 13 | 77 |
| [SuperScaler/src/backend/common_runtime/executor/task_sched.cpp](/SuperScaler/src/backend/common_runtime/executor/task_sched.cpp) | C++ | 124 | 29 | 31 | 184 |
| [SuperScaler/src/backend/common_runtime/executor/task_sched.hpp](/SuperScaler/src/backend/common_runtime/executor/task_sched.hpp) | C++ | 36 | 49 | 16 | 101 |
| [SuperScaler/src/backend/common_runtime/executor/test/channel/test_cpu_channel.cpp](/SuperScaler/src/backend/common_runtime/executor/test/channel/test_cpu_channel.cpp) | C++ | 92 | 7 | 9 | 108 |
| [SuperScaler/src/backend/common_runtime/executor/test/cuda_ipc/test_cuda_channel.cpp](/SuperScaler/src/backend/common_runtime/executor/test/cuda_ipc/test_cuda_channel.cpp) | C++ | 74 | 11 | 10 | 95 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_poll_executor.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_poll_executor.cpp) | C++ | 96 | 13 | 22 | 131 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_reduction_task.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_reduction_task.cpp) | C++ | 340 | 2 | 76 | 418 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_ring_buffer.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_ring_buffer.cpp) | C++ | 207 | 20 | 13 | 240 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_scale_task.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_scale_task.cpp) | C++ | 227 | 2 | 54 | 283 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_semaphore.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_semaphore.cpp) | C++ | 102 | 18 | 11 | 131 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_send_recv_task.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_send_recv_task.cpp) | C++ | 98 | 3 | 11 | 112 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_shared_memory.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_shared_memory.cpp) | C++ | 57 | 5 | 8 | 70 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_task.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_task.cpp) | C++ | 43 | 3 | 8 | 54 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_task_manager.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_task_manager.cpp) | C++ | 113 | 8 | 23 | 144 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_task_sched.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_task_sched.cpp) | C++ | 174 | 25 | 35 | 234 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_thread_safe_queue.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_thread_safe_queue.cpp) | C++ | 58 | 2 | 16 | 76 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_worker.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_worker.cpp) | C++ | 69 | 2 | 9 | 80 |
| [SuperScaler/src/backend/common_runtime/executor/test/test_worker_sched.cpp](/SuperScaler/src/backend/common_runtime/executor/test/test_worker_sched.cpp) | C++ | 68 | 3 | 18 | 89 |
| [SuperScaler/src/backend/common_runtime/executor/test/utils/utils.cpp](/SuperScaler/src/backend/common_runtime/executor/test/utils/utils.cpp) | C++ | 12 | 2 | 3 | 17 |
| [SuperScaler/src/backend/common_runtime/executor/test/utils/utils.hpp](/SuperScaler/src/backend/common_runtime/executor/test/utils/utils.hpp) | C++ | 4 | 8 | 2 | 14 |
| [SuperScaler/src/backend/common_runtime/executor/utils/ring_buffer.cpp](/SuperScaler/src/backend/common_runtime/executor/utils/ring_buffer.cpp) | C++ | 69 | 5 | 10 | 84 |
| [SuperScaler/src/backend/common_runtime/executor/utils/ring_buffer.hpp](/SuperScaler/src/backend/common_runtime/executor/utils/ring_buffer.hpp) | C++ | 58 | 32 | 16 | 106 |
| [SuperScaler/src/backend/common_runtime/executor/utils/semaphore_wrapper.cpp](/SuperScaler/src/backend/common_runtime/executor/utils/semaphore_wrapper.cpp) | C++ | 120 | 11 | 14 | 145 |
| [SuperScaler/src/backend/common_runtime/executor/utils/semaphore_wrapper.hpp](/SuperScaler/src/backend/common_runtime/executor/utils/semaphore_wrapper.hpp) | C++ | 28 | 26 | 10 | 64 |
| [SuperScaler/src/backend/common_runtime/executor/utils/shared_memory.cpp](/SuperScaler/src/backend/common_runtime/executor/utils/shared_memory.cpp) | C++ | 106 | 9 | 10 | 125 |
| [SuperScaler/src/backend/common_runtime/executor/utils/shared_memory.hpp](/SuperScaler/src/backend/common_runtime/executor/utils/shared_memory.hpp) | C++ | 20 | 9 | 6 | 35 |
| [SuperScaler/src/backend/common_runtime/executor/utils/thread_safe_queue.hpp](/SuperScaler/src/backend/common_runtime/executor/utils/thread_safe_queue.hpp) | C++ | 49 | 22 | 9 | 80 |
| [SuperScaler/src/backend/common_runtime/executor/worker.cpp](/SuperScaler/src/backend/common_runtime/executor/worker.cpp) | C++ | 77 | 3 | 13 | 93 |
| [SuperScaler/src/backend/common_runtime/executor/worker.hpp](/SuperScaler/src/backend/common_runtime/executor/worker.hpp) | C++ | 37 | 2 | 12 | 51 |
| [SuperScaler/src/backend/common_runtime/executor/worker_sched.cpp](/SuperScaler/src/backend/common_runtime/executor/worker_sched.cpp) | C++ | 80 | 6 | 13 | 99 |
| [SuperScaler/src/backend/common_runtime/executor/worker_sched.hpp](/SuperScaler/src/backend/common_runtime/executor/worker_sched.hpp) | C++ | 31 | 16 | 14 | 61 |
| [SuperScaler/src/backend/common_runtime/session.cpp](/SuperScaler/src/backend/common_runtime/session.cpp) | C++ | 232 | 14 | 28 | 274 |
| [SuperScaler/src/backend/common_runtime/session.hpp](/SuperScaler/src/backend/common_runtime/session.hpp) | C++ | 51 | 5 | 10 | 66 |
| [SuperScaler/src/backend/common_runtime/test/plan/0/plan.json](/SuperScaler/src/backend/common_runtime/test/plan/0/plan.json) | JSON | 7,776 | 0 | 0 | 7,776 |
| [SuperScaler/src/backend/common_runtime/test/plan/1/plan.json](/SuperScaler/src/backend/common_runtime/test/plan/1/plan.json) | JSON | 7,776 | 0 | 0 | 7,776 |
| [SuperScaler/src/backend/common_runtime/test/superscaler_op_test.cc](/SuperScaler/src/backend/common_runtime/test/superscaler_op_test.cc) | C++ | 329 | 2 | 22 | 353 |
| [SuperScaler/src/backend/common_runtime/test/superscaler_rt_test.cc](/SuperScaler/src/backend/common_runtime/test/superscaler_rt_test.cc) | C++ | 169 | 1 | 28 | 198 |
| [SuperScaler/src/backend/common_runtime/util.cpp](/SuperScaler/src/backend/common_runtime/util.cpp) | C++ | 64 | 4 | 8 | 76 |
| [SuperScaler/src/backend/common_runtime/util.hpp](/SuperScaler/src/backend/common_runtime/util.hpp) | C++ | 71 | 2 | 16 | 89 |
| [SuperScaler/src/backend/superscaler_pywrap.cpp](/SuperScaler/src/backend/superscaler_pywrap.cpp) | C++ | 41 | 4 | 12 | 57 |
| [SuperScaler/src/backend/superscaler_pywrap.hpp](/SuperScaler/src/backend/superscaler_pywrap.hpp) | C++ | 16 | 10 | 10 | 36 |
| [SuperScaler/src/backend/tensorflow/gpu_util.hpp](/SuperScaler/src/backend/tensorflow/gpu_util.hpp) | C++ | 58 | 34 | 16 | 108 |
| [SuperScaler/src/backend/tensorflow/operation.cpp](/SuperScaler/src/backend/tensorflow/operation.cpp) | C++ | 254 | 18 | 43 | 315 |
| [SuperScaler/src/backend/tensorflow/operation.hpp](/SuperScaler/src/backend/tensorflow/operation.hpp) | C++ | 16 | 10 | 10 | 36 |
| [SuperScaler/src/superscaler/ai_simulator/README.md](/SuperScaler/src/superscaler/ai_simulator/README.md) | Markdown | 54 | 0 | 30 | 84 |
| [SuperScaler/src/superscaler/ai_simulator/__init__.py](/SuperScaler/src/superscaler/ai_simulator/__init__.py) | Python | 2 | 2 | 3 | 7 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/__init__.py](/SuperScaler/src/superscaler/ai_simulator/simulator/__init__.py) | Python | 2 | 2 | 3 | 7 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/computation_device.py](/SuperScaler/src/superscaler/ai_simulator/simulator/computation_device.py) | Python | 31 | 2 | 12 | 45 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/device.py](/SuperScaler/src/superscaler/ai_simulator/simulator/device.py) | Python | 21 | 10 | 9 | 40 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/device_factory.py](/SuperScaler/src/superscaler/ai_simulator/simulator/device_factory.py) | Python | 18 | 4 | 8 | 30 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/fifo_device.py](/SuperScaler/src/superscaler/ai_simulator/simulator/fifo_device.py) | Python | 20 | 10 | 8 | 38 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/__init__.py](/SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/flow.py](/SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/flow.py) | Python | 68 | 6 | 14 | 88 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/link.py](/SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/link.py) | Python | 46 | 4 | 13 | 63 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/link_manager.py](/SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/link_manager.py) | Python | 101 | 15 | 15 | 131 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/network_simulator.py](/SuperScaler/src/superscaler/ai_simulator/simulator/network_simulator/network_simulator.py) | Python | 202 | 35 | 23 | 260 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/node.py](/SuperScaler/src/superscaler/ai_simulator/simulator/node.py) | Python | 141 | 40 | 44 | 225 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/simulator.py](/SuperScaler/src/superscaler/ai_simulator/simulator/simulator.py) | Python | 267 | 28 | 46 | 341 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/tensor.py](/SuperScaler/src/superscaler/ai_simulator/simulator/tensor.py) | Python | 61 | 5 | 12 | 78 |
| [SuperScaler/src/superscaler/ai_simulator/simulator/utility.py](/SuperScaler/src/superscaler/ai_simulator/simulator/utility.py) | Python | 26 | 6 | 8 | 40 |
| [SuperScaler/src/superscaler/plan_gen/__init__.py](/SuperScaler/src/superscaler/plan_gen/__init__.py) | Python | 16 | 4 | 8 | 28 |
| [SuperScaler/src/superscaler/plan_gen/plan/__init__.py](/SuperScaler/src/superscaler/plan_gen/plan/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/plan_gen/plan/adapter/__init__.py](/SuperScaler/src/superscaler/plan_gen/plan/adapter/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/plan_gen/plan/adapter/adapter.py](/SuperScaler/src/superscaler/plan_gen/plan/adapter/adapter.py) | Python | 3 | 2 | 2 | 7 |
| [SuperScaler/src/superscaler/plan_gen/plan/adapter/ai_simulator_adapter.py](/SuperScaler/src/superscaler/plan_gen/plan/adapter/ai_simulator_adapter.py) | Python | 161 | 40 | 28 | 229 |
| [SuperScaler/src/superscaler/plan_gen/plan/allreduce_plan.py](/SuperScaler/src/superscaler/plan_gen/plan/allreduce_plan.py) | Python | 73 | 7 | 15 | 95 |
| [SuperScaler/src/superscaler/plan_gen/plan/node_list.py](/SuperScaler/src/superscaler/plan_gen/plan/node_list.py) | Python | 117 | 2 | 19 | 138 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/DAG_parser.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/DAG_parser.py) | Python | 8 | 8 | 7 | 23 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/__init__.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/import json.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/import%20json.py) | Python | 39 | 4 | 14 | 57 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/__init__.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/database_backend.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/database_backend.py) | Python | 67 | 8 | 19 | 94 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/database_loader.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/database_loader.py) | Python | 139 | 2 | 17 | 158 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/profiler.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/profiler/profiler.py) | Python | 35 | 4 | 9 | 48 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/tf_parser.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/tf_parser.py) | Python | 503 | 44 | 80 | 627 |
| [SuperScaler/src/superscaler/plan_gen/plan/parser/torch_parser.py](/SuperScaler/src/superscaler/plan_gen/plan/parser/torch_parser.py) | Python | 77 | 21 | 19 | 117 |
| [SuperScaler/src/superscaler/plan_gen/plan/plan.py](/SuperScaler/src/superscaler/plan_gen/plan/plan.py) | Python | 94 | 6 | 17 | 117 |
| [SuperScaler/src/superscaler/plan_gen/plan/plan_generator.py](/SuperScaler/src/superscaler/plan_gen/plan/plan_generator.py) | Python | 53 | 14 | 18 | 85 |
| [SuperScaler/src/superscaler/plan_gen/plan/plan_manager.py](/SuperScaler/src/superscaler/plan_gen/plan/plan_manager.py) | Python | 29 | 12 | 8 | 49 |
| [SuperScaler/src/superscaler/plan_gen/plan/plan_mapper.py](/SuperScaler/src/superscaler/plan_gen/plan/plan_mapper.py) | Python | 81 | 17 | 22 | 120 |
| [SuperScaler/src/superscaler/plan_gen/plan/plan_pool.py](/SuperScaler/src/superscaler/plan_gen/plan/plan_pool.py) | Python | 63 | 2 | 12 | 77 |
| [SuperScaler/src/superscaler/plan_gen/plan/raw_allreduce_plan.py](/SuperScaler/src/superscaler/plan_gen/plan/raw_allreduce_plan.py) | Python | 6 | 2 | 7 | 15 |
| [SuperScaler/src/superscaler/plan_gen/plan/recursive_halving_plan.py](/SuperScaler/src/superscaler/plan_gen/plan/recursive_halving_plan.py) | Python | 153 | 35 | 33 | 221 |
| [SuperScaler/src/superscaler/plan_gen/plan/reduce_broadcast_allreduce_plan.py](/SuperScaler/src/superscaler/plan_gen/plan/reduce_broadcast_allreduce_plan.py) | Python | 85 | 16 | 18 | 119 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/__init__.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/hardware.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/hardware.py) | Python | 132 | 9 | 29 | 170 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/link.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/link.py) | Python | 73 | 4 | 18 | 95 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/resource.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/resource.py) | Python | 9 | 2 | 6 | 17 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/resource_pool.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/resource_pool.py) | Python | 221 | 17 | 39 | 277 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/router.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/router.py) | Python | 160 | 20 | 29 | 209 |
| [SuperScaler/src/superscaler/plan_gen/plan/resources/server.py](/SuperScaler/src/superscaler/plan_gen/plan/resources/server.py) | Python | 60 | 3 | 13 | 76 |
| [SuperScaler/src/superscaler/plan_gen/plan/ring_allreduce_plan.py](/SuperScaler/src/superscaler/plan_gen/plan/ring_allreduce_plan.py) | Python | 86 | 23 | 17 | 126 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)