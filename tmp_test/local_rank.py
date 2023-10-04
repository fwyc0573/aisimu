import argparse
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
print(f"args: {args}")

for key, value in os.environ.items():
    print(key, "=", value)

# rank = int(os.environ['SLURM_PROCID'])
# local_rank = int(os.environ['SLURM_LOCALID'])
# world_size = int(os.environ['SLURM_NTASKS'])

# print("rank: ", rank)
# print("local_rank: ", local_rank)
# print("world_size: ", world_size)
