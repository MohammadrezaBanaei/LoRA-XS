import warnings
import torch
import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def merge_latent(self):
    if self.active_adapter[0] not in self.lora_A.keys():
        return
    if self.merged:
        warnings.warn("Already merged. Nothing to do.")
        return
    if self.r[self.active_adapter[0]] > 0:
        # adding latent_mapping in the forward loop
        self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter[0]].weight @ self.default_lora_latent_mapping.weight @ self.lora_A[self.active_adapter[0]].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter[0]]
        )
        self.merged = True


def forward_latent(self, x: torch.Tensor):
    previous_dtype = x.dtype

    if self.active_adapter[0] not in self.lora_A.keys():
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    if self.disable_adapters:
        if self.r[self.active_adapter[0]] > 0 and self.merged:
            self.unmerge()
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    elif self.r[self.active_adapter[0]] > 0 and not self.merged:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)

        # adding latent_mapping in the forward loop
        result += (
            self.lora_B[self.active_adapter[0]](
                self.default_lora_latent_mapping(
                    self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x))
                )
            )
            * self.scaling[self.active_adapter[0]]
        )
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    result = result.to(previous_dtype)

    return result

