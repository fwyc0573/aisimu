
NNRANK=0 

NNODES=$1
model=$2
batchsize=$3
type=$4

# FIXME: 路径改一下
export PYTHONPATH=/mnt/aisim/SuperScaler/src/ 
# 在程序启动前加载特定的共享库，这里加载了NCCL的网络插件，这可能是为了提升RDMA通信性能
export LD_PRELOAD=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
# 设置共享库的搜索路径，包括Mellanox的SHARP库和NCCL插件库，这些可能是为了提高网络通信的性能
export LD_LIBRARY_PATH=/opt2/mellanox/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib
# 配置了NCCL使用UCX通信框架的细节
export NCCL_UCX_TLS=rc_x,cuda_copy,cuda_ipc
export NCCL_UCX_RNDV_THRESH=0
export NCCL_UCX_RNDV_SCHEME=get_zcopy
# 设置UCX通信框架的性能相关参数，比如启用多线程和禁用内存类型缓存
export UCX_RC_MLX5_TM_ENABLE=y
export UCX_MEMTYPE_CACHE=n
# 开启IB网络接口的PCI宽松排序，可能用于优化内存访问顺序
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
# 设置GPU Direct RDMA的级别，用于GPU之间的直接通信
export NCCL_NET_GDR_LEVEL=5
# 确保CUDA设备按照PCI总线ID顺序枚举，这有助于确定性地选择GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 设置NCCL使用的网络接口名称
export NCCL_SOCKET_IFNAME=eth0
# 设置点对点通信插件为IB（InfiniBand）
export NCCL_PLUGIN_P2P=ib
# 指定NCCL的拓扑配置文件，用于优化通信路径。
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
# 设置NCCL的调试子系统，这里设置为仅初始化过程
export NCCL_DEBUG_SUBSYS=INIT
# 设置AR（Adaptive Routing）阈值，影响InfiniBand网络的路由选择
export NCCL_IB_AR_THRESHOLD=0

# --nproc_per_node: 每个节点的GPU数量
# --nnodes 参与训练的节点总数
# --node_rank 设置当前节点的编号，这里固定为0，表示这是主节点
# --master_addr: 设置主节点地址，用于所有节点之间的通信
# --model batchsize type 皆是传入py文件的参数
python3 -m torch.distributed.launch --nproc_per_node 8 \
    --nnodes $NNODES \
    --node_rank $NNRANK \
    --master_addr msrhpc-msccl-000000 \
    ddp_profile.py \
    --model $model \
    --batchsize $batchsize \
    --type $type
