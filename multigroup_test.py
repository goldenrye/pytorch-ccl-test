# torchrun --nproc-per-node=2 --nnodes=1 --rdzv-id=my_job_id --rdzv-backend=c10d --rdzv-endpoint=localhost:12345 multigroup_test.py
import torch
import torch.distributed as dist
import os

def setup_process_group(backend, rank, world_size, group_name):
    dist.init_process_group(backend=backend, init_method='env://', world_size=world_size, rank=rank, group_name=group_name)

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # Initialize the process group for Gloo
    setup_process_group('gloo', rank, world_size, 'gloo_group')  # Adjust rank and world_size as needed
    gloo_group = dist.new_group(backend='gloo')

    # Prepare data for Gloo
    data_gloo = torch.tensor([dist.get_rank()], dtype=torch.float32)
    
    # Perform all_gather_object with Gloo
    gathered_gloo = [torch.zeros_like(data_gloo) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_gloo, data_gloo, group=gloo_group)
    
    print(f"Gloo Group - Rank {dist.get_rank()} gathered: {gathered_gloo}")

    # Initialize the process group for NCCL
    # setup_process_group(backend='nccl', rank=0, world_size=2, group_name='nccl_group')  # Adjust rank and world_size as needed
    nccl_group = dist.new_group(backend='nccl')

    # Prepare data for NCCL
    torch.cuda.set_device(dist.get_rank())
    data_nccl = torch.ones((1024, 1024), device=f"cuda:{dist.get_rank()}")
    
    # Perform all_gather_object with NCCL
    gathered_nccl = [torch.zeros_like(data_nccl) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_nccl, data_nccl, group=nccl_group)

    print(f"NCCL Group - Rank {dist.get_rank()} gathered: {gathered_nccl}")
if __name__ == "__main__":
    main()

