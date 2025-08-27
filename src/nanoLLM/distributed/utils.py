import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def get_rank():
    return dist.get_rank()

def get_world_size():
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def reduce_tensor(tensor, op='mean'):
    """Reduce tensor across all processes"""
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if op == 'mean':
        tensor /= get_world_size()
    return tensor

def barrier():
    """Synchronize all processes"""
    dist.barrier()