# torchrun --nproc_per_node=4 allgather_test.py

import torch
import torch.distributed as dist
import os
import sys

def test_all_gather_object():
    # Get the global rank and local rank from environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    
    # Set the device for the current process
    torch.cuda.set_device(local_rank)

    # Create a sample object to gather
    local_data = f"Data from rank {global_rank}:{local_rank}"

    # Prepare the output list
    world_size = dist.get_world_size()
    gathered_data = [None] * world_size

    # Perform all_gather_object
    dist.all_gather_object(gathered_data, local_data)

    # Print the results from rank 0
    if global_rank == 0:
        print(f"Gathered data: {gathered_data}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    # Run the test
    test_all_gather_object()
    
    # Force exit to terminate all processes
    sys.exit(0)

