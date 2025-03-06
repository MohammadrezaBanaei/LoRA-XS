import math
import types

import peft
import torch
from peft.import_utils import is_bnb_available
from peft.utils import _get_submodules
from torch.nn import init
from tqdm import tqdm

from .latent_utils import get_delta_weight, forward_latent


def get_svd_replacement_module(weight, cfg):
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    u = u @ torch.diag(s)
    r = cfg['rank']
    retain_part = cfg['retain_part']  # 'top' or 'bottom'

    if retain_part == 'top':
        u = u[:, :r]  # Select top r singular vectors
        vh = vh[:r, :]
    elif retain_part == 'bottom':
        u = u[:, -r:]  # Select bottom r singular vectors
        vh = vh[-r:, :]
    else:
        raise ValueError(f"Invalid retain_part: {retain_part}, should be 'top' or 'bottom'")

    return u, vh


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(target_module, new_weight):
    device = target_module.weight.device
    target_module.weight = torch.nn.Parameter(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)


def update_decoder_weights(target_module, new_weight):
    device = target_module.weight.device
    with torch.no_grad():
        target_module.weight.copy_(new_weight)

    # dispatch to correct device
    for name, module in target_module.named_modules():
        if "lora_" in name:
            module.to(device)

def kaiming_uniform_init(matrix: torch.tensor):
    init.kaiming_uniform_(matrix, a=math.sqrt(5))
    return matrix


def find_and_initialize(model, peft_config, adapter_name, reconstruct_config):
    """
    :param adapter_name: options: 'default'
    """
    lora_config = peft_config[adapter_name]

    init_type = reconstruct_config['init_type']
    assert init_type in ['svd_on_w', 'random', 'svd_on_random']
    is_r_squared = reconstruct_config['r_squared']
    print('Config:', reconstruct_config)

    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    if loaded_in_8bit and not is_bnb_available():
        raise ImportError(
            "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
            "You can install it with `pip install bitsandbytes`."
        )
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))

    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, target_name = _get_submodules(model, key)

            if init_type == 'svd_on_w':
                replacement_encoder_weight, replacement_decoder_weight = get_svd_replacement_module(weight=target.weight.T,
                                                                                                    cfg=reconstruct_config)
            elif init_type == 'svd_on_random':
                random_w = kaiming_uniform_init(torch.zeros_like(target.weight.T))
                replacement_encoder_weight, replacement_decoder_weight = get_svd_replacement_module(weight=random_w,
                                                                                                    cfg=reconstruct_config)
            elif init_type == 'random':
                random_w = kaiming_uniform_init(torch.zeros_like(target.weight.T))
                replacement_encoder_weight, replacement_decoder_weight = get_svd_replacement_module(weight=random_w,
                                                                                                    cfg=reconstruct_config)
                kaiming_uniform_init(replacement_encoder_weight)
                kaiming_uniform_init(replacement_decoder_weight)

            if not isinstance(target, peft.tuners.lora.Linear):
                raise NotImplementedError('Only initialization for peft.tuners.lora.Linear type is implemented.')
            else:
                replace_module_weights(target.lora_B.default, replacement_decoder_weight.T)
                target.forward = types.MethodType(forward_latent, target)
                target.get_delta_weight = types.MethodType(get_delta_weight, target)
                replace_module_weights(target.lora_A.default, replacement_encoder_weight.T)
                target.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                init_module_weights(target.default_lora_latent_mapping, sigma=0.00001)
                target.default_lora_latent_mapping.to(target.lora_A.default.weight.device)

                target.lora_A.default.weight.requires_grad = False  # only the r*r matrix will be tuned
                target.lora_B.default.weight.requires_grad = False  # only the r*r matrix will be tuned

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
