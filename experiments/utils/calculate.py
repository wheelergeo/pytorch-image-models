from typing import Callable
from functools import wraps
from thop import profile, clever_format


import time
import warnings
import logging
import torch


def calculate_inference_time(verbosity: bool=False):
    """
    Decorator which calculate the inference time in seconds.

    Args:
        verbosity (bool): If False, ignore warnings and useless logging messages.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not verbosity:
                warnings.filterwarnings("ignore")
                logging.getLogger("transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            start_time = time.time()
            outputs = func(*args, **kwargs)
            torch.cuda.synchronize()  # Wait for all GPU tasks to complete
            end_time = time.time()
            return end_time - start_time, outputs
        return wrapper
    return decorator


def calculate_flops(model, batch_size: int = 1, sys_token_len: int = 15, 
                    seq_len: int = 354, vision_token_len: int = 324):
    """
        input_ids: torch.Size([1, 354])
        inputs_embeds: torch.Size([1, 354, 3584])
        pixel_values: torch.Size([1296, 1176])
        attention_mask: torch.Size([1, 354])
        image_embeds: torch.Size([324, 3584])
        image_grid_thw: tensor([[ 1, 36, 36]], device='cuda:0')
        inputs_embeds.shape: torch.Size([1, 62, 3584])
        position_ids.shape: torch.Size([4, 1, 62])
        cache_position.shape: torch.Size([62])
        attention_mask.shape: torch.Size([1, 62])
    """
    input_ids = torch.cat((
        torch.arange(sys_token_len, device=model.device), 
        torch.full((vision_token_len,), model.config.image_token_id, device=model.device),
        torch.arange(seq_len - sys_token_len - vision_token_len, device=model.device)
    )).unsqueeze(0).repeat(batch_size, 1)

    dummy_inputs = (
        input_ids,
        torch.ones((batch_size, seq_len), dtype=torch.long).to(model.device),
        torch.arange(seq_len, device=model.device).view(1, 1, seq_len).expand(4, batch_size, seq_len),
        None,
        None,
        None,
        True,
        False,
        False,
        torch.rand((1296, 1176), dtype=torch.float32, device=model.device) * 2 - 1,
        None,
        torch.tensor([[1, 36, 36]], dtype=torch.long).to(model.device),
        None,
        None,
        torch.arange(seq_len, device=model.device),
    )

    flops, params = profile(model, inputs=dummy_inputs, verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    
    return flops, params