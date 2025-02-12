# this script is for gpu tensor all_gather test using gloo backend
# 2 nodes: torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=job_1234 --rdzv_backend=c10d --rdzv_endpoint=fdbd:dc61:20:11::16:12345 ./allgather_cpu.py

import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def test_all_gather():
    """Test the all_gather functionality."""
    # Initialize distributed environment
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Create a tensor specific to each rank without moving it to GPU
    local_tensor = torch.tensor([rank], dtype=torch.float32)

    # Prepare an empty list of tensors to gather data from all ranks
    gathered_tensors = [torch.empty_like(local_tensor) for _ in range(world_size)]

    # Perform all_gather operation
    dist.all_gather(gathered_tensors, local_tensor)

    # Print gathered tensors from each rank
    print(f"Rank {rank}: Gathered tensors: {gathered_tensors}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    test_all_gather()

