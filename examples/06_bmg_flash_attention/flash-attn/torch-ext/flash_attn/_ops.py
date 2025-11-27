import torch
from . import _flash_attn_test
ops = torch.ops._flash_attn_test

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_flash_attn_test::{op_name}"