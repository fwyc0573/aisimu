import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# 通过命令行传入参数
parser = argparse.ArgumentParser()
parser.add_argument("--world_size", type=int)
parser.add_argument("--node_rank", type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="12355", type=str)
args = parser.parse_args()

# TODO：1
def example(local_rank, node_rank, local_size, world_size):
    # 初始化
    rank = local_rank + node_rank * local_size
    print(f"1. rank: {rank}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl",
                            init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
                            rank=rank,
                            world_size=world_size)
    print(f"2. rank: {rank} init success")
    

    # ---------------- 定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做 ---------------- #
    # model = Model()
    # model.to(device)
    # ---------------------------------------------------------------------------- #

    # 创建模型
    model = nn.Linear(10, 10).to(local_rank)

    # 放入DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank) 
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(f"3. rank: {rank} ddp success")

    # 进行前向后向计算
    print("start training...")
    for i in range(1000):
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 10).to(local_rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
    print("finish training...")


def main():
    local_size = torch.cuda.device_count()
    print(f"local_size: {local_size}" )

    mp.spawn(example,
        args=(args.node_rank, local_size, args.world_size,),
        nprocs=local_size,
        join=True)



if __name__=="__main__":
    main()


# ----------------------------------- 单机多卡 ----------------------------------- #
# >>> #节点1
# >>> python multiple.py --world_size=2 --node_rank=0 --master_addr="172.17.0.3" --master_port=22335


# ----------------------------------- 多机多卡 ----------------------------------- #
# >>> #节点1
# >>> python multiple.py --world_size=4 --node_rank=0 --master_addr="172.17.0.3" --master_port=22335

# >>> #节点2
# >>> python multiple.py --world_size=4 --node_rank=1 --master_addr="172.17.0.3" --master_port=22335
