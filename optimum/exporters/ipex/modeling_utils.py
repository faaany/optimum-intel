#  Copyright 2024 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import gc
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv

from optimum.intel.utils.import_utils import is_ipex_version

import intel_extension_for_pytorch as ipex


def matmul_add_add(attn_output, weight, bias=None, residual=None):
    seq_len, bs, _ = attn_output.size()
    if residual is None:
        attn_output = torch.matmul(attn_output, weight)
        if bias is not None:
            attn_output += bias
    else:
        if bias is not None:
            attn_output = torch.ops.torch_ipex.mm_bias_resadd(attn_output, weight, bias, 1.0, residual, 1.0)
        else:
            attn_output = torch.addmm(
                residual.flatten(0, -2),
                attn_output.flatten(0, -2),
                weight,
                beta=1.0,
            )
    attn_output = attn_output.view(seq_len, bs, -1)
    return attn_output

def reference_elimination(c, b):
    for item in gc.get_objects():
        if isinstance(item, torch.Tensor) and item.data_ptr() == c.data_ptr() and item is not c:
            item.data = b
            
# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
def _llama_layer_norm_forward(self, hidden_states):
    return torch.ops.torch_ipex.rmsnorm(hidden_states, self.weight, self.variance_epsilon)


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L321
def _llama_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    kv_seq_len = q_len + past_key_value[0].size(-2) if past_key_value is not None else q_len

    query = query.view(bsz, q_len, self.num_heads, self.head_dim)
    key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    # Use ipex op to rotary position embedding more efficient.
    key = self.ipex_rope(
        key,
        position_ids,
        self.num_key_value_heads,
        self.head_dim,
        self.head_dim // 2,
        self.head_dim,
        kv_seq_len,
    )
    query = self.ipex_rope(
        query,
        position_ids,
        self.num_heads,
        self.head_dim,
        self.head_dim // 2,
        self.head_dim,
        kv_seq_len,
    )

    if use_cache:
        # This ipex op pre-allocates buffers for past_key_values and use beam index history
        # which to decide which beam should be used to make attention scale dot more efficient.
        (attn_output, attn_weights, past_key_value) = self.ipex_scale_dot_product(
            query,
            key,
            value,
            math.sqrt(self.head_dim),
            past_key_value,
            None,
            attention_mask,
        )
    else:
        value_states = value.transpose(1, 2)
        query_states = query.transpose(1, 2)
        key_states = key.transpose(1, 2)
        kv_seq_len = key_states.shape[-2]

        past_key_value = None
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = torch.tensor(attn_weights) + torch.tensor(attention_mask)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if not output_attentions:
        attn_weights = None

    return attn_output, past_key_value, attn_weights


def padding_attn_mask(attn_mask, alignment):
    if attn_mask is None:
        return None
    assert isinstance(
        attn_mask, torch.Tensor
    ), f"attn mask is supposed to be a tensor, instead we got {type(attn_mask)}"
    if attn_mask.device == torch.device("cpu"):
        return attn_mask
    last_dim_size = attn_mask.size(-1)
    aligned_size = (last_dim_size + alignment - 1) // alignment * alignment
    mask_size = [*attn_mask.size()[:-1], aligned_size]
    new_attn_mask = torch.empty(mask_size, dtype=attn_mask.dtype, device=attn_mask.device).fill_(-65504.0)
    new_attn_mask[..., :last_dim_size] = attn_mask
    return new_attn_mask


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L1130
def _llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    attention_mask = padding_attn_mask(attention_mask, 8)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    seqlen = hidden_states.size(1)
    head_dim = self.layers[0].attn.head_dim
    sin, cos = self.layers[0].attn.ipex_rope.get_sin_cos(seqlen, head_dim // 2)
    sin = sin.squeeze()[position_ids].unsqueeze(2)
    cos = cos.squeeze()[position_ids].unsqueeze(2)
    sin_cos = {"sin": sin, "cos": cos}
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None and len(past_key_values) > idx else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **sin_cos,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayerRef(nn.Module):
    def __init__(self, module, config, distributed=False):
        if is_ipex_version("<", "2.5.0"):
            raise ImportError("Only ipex version > 2.3.0 supports Linear2SiluMul and LinearAdd")

        from intel_extension_for_pytorch.llm.modules import Linear2SiluMul, LinearAdd

        super().__init__()
        for k, v in module.__dict__.items():
            setattr(self, k, v)
        for k, v in module.__class__.__dict__.items():
            if k.startswith("__") or k.startswith("forward"):
                continue
            setattr(self.__class__, k, getattr(module.__class__, k))
        self.distributed = distributed
        if not self.distributed:
            self.mha_linear_add = LinearAdd(module.self_attn.o_proj)
            self.mlp_linear_add = LinearAdd(module.mlp.down_proj)
            del self.__dict__["_modules"]["self_attn"].o_proj
            del self.__dict__["_modules"]["mlp"].down_proj
        self.linear_silu_mul = Linear2SiluMul(module.mlp.gate_proj, module.mlp.up_proj)
        del self.__dict__["_modules"]["mlp"].gate_proj
        del self.__dict__["_modules"]["mlp"].up_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if not self.distributed:
            hidden_states = self.mha_linear_add(hidden_states, residual)
        else:
            hidden_states = self.self_attn.o_proj(hidden_states)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_gate = self.linear_silu_mul(hidden_states)

        if not self.distributed:
            hidden_states = self.mlp_linear_add(mlp_gate, residual)
        else:
            hidden_states = self.mlp.down_proj(mlp_gate)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
            
            
class _IPEXLlamaAttention(nn.Module):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.distributed = distributed
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_kv_heads = module.num_key_value_heads
        self.embed_dim = module.config.hidden_size
        module_device = str(module.q_proj.weight.device)
        if "xpu" in module_device:
            self.ipex_scale_dot_product = None
            
            from intel_extension_for_pytorch.transformers.models.xpu.fusions.mha_fusion import _IPEXRopeXPU
            self.ipex_rope = _IPEXRopeXPU(
                module.config.max_position_embeddings,
                module.config.hidden_size // module.config.num_attention_heads,
                module.config.rope_theta,
                module.config.architectures[0],
            )
            self.port_parameters(module)
            torch.xpu.empty_cache()
            
        else:
            from intel_extension_for_pytorch.llm.modules import IndirectAccessKVCache
            from intel_extension_for_pytorch.llm.modules import RotaryEmbedding
            self.ipex_scale_dot_product = IndirectAccessKVCache(text_max_length=module.config.max_position_embeddings)
            
            self.ipex_rope = RotaryEmbedding(
            module.config.max_position_embeddings,
            module.config.hidden_size // module.config.num_attention_heads,
            module.config.rope_theta,
            module.config.architectures[0],
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            residual (`torch.Tensor`): residual tensor to the layer of shape `
        """
        # allocate cache and copy past_key_value
        bs, seqlen, _ = hidden_states.size()
        prev_seqlen = 0
        if past_key_value:
            _, _, prev_seqlen, _ = past_key_value[0].size()
        if self.num_kv_heads == self.num_heads:
            query = torch.empty(
                (bs, seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            key = torch.empty(
                (bs, prev_seqlen + seqlen, self.num_heads * self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            value = torch.empty(
                (bs, prev_seqlen + seqlen, self.num_heads * self.head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.torch_ipex.mm_qkv_out(
                hidden_states,
                self.qkv_proj_weight,
                self.qkv_proj_bias,
                query,
                key[:, prev_seqlen:, :],
                value[:, prev_seqlen:, :],
            )
        else:
            query = torch.empty(
                (bs, seqlen, self.num_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            key = torch.empty(
                (bs, seqlen, self.num_kv_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            value = torch.empty(
                (bs, seqlen, self.num_kv_heads * self.head_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )
            torch.ops.torch_ipex.mm_qkv_group_out(
                hidden_states, self.qkv_proj_weight, self.qkv_proj_bias, query, key, value
            )

        query = query.view([bs, seqlen, self.num_heads, self.head_dim])
        key = key.view([bs, seqlen + prev_seqlen, self.num_kv_heads, self.head_dim])

        if hasattr(kwargs, "sin") and hasattr(kwargs, "cos"):
            print("cache sin cos")
            sin = kwargs["sin"]
            cos = kwargs["cos"]
        else:
            sin, cos = self.ipex_rope.get_sin_cos(seqlen, self.head_dim // 2)
            sin = sin.squeeze()[position_ids].unsqueeze(2)
            cos = cos.squeeze()[position_ids].unsqueeze(2)
        self.ipex_rope.apply_embedding(query, sin, cos, self.head_dim // 2, key[:, prev_seqlen:, :, :])
        value = value.view([bs, seqlen + prev_seqlen, self.num_kv_heads, self.head_dim])
        if past_key_value is not None:
            value[:, :prev_seqlen, :, :] = past_key_value[1].transpose(1, 2)
            key[:, :prev_seqlen, :, :] = past_key_value[0].transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        present = (key, value) if use_cache else None

        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = False
        attn_output = torch.xpu.IpexSDP(
            query, key, value, None, attention_mask, None, scale, 1.0, 0.0, is_causal, False
        )
        attn_output = attn_output.transpose(1, 2).view([bs, seqlen, self.embed_dim])
        attn_output = matmul_add_add(attn_output, self.o_proj_weight, self.o_proj_bias, residual).view(
            [bs, seqlen, self.embed_dim]
        )
        outputs = (attn_output, present)
        if output_attentions:
            raise ValueError("not support output attn_weight")
        else:
            outputs += (None,)
        return outputs

    def port_parameters(self, module):
        self.qkv_proj_bias = None
        self.qkv_proj_weight = None
        if self.num_heads == self.num_kv_heads:
            q_proj = module.q_proj.weight.transpose(0, 1)
            k_proj = module.k_proj.weight.transpose(0, 1)
            v_proj = module.v_proj.weight.transpose(0, 1)
            self.qkv_proj_weight = torch.stack([q_proj, k_proj, v_proj]).contiguous().view([3, -1, q_proj.shape[-1]])
            reference_elimination(module.q_proj.weight.data, self.qkv_proj_weight[0, :, :].transpose(0, 1))
            module.q_proj.weight.data = self.qkv_proj_weight[0, :, :].transpose(0, 1)
            reference_elimination(module.k_proj.weight.data, self.qkv_proj_weight[1, :, :].transpose(0, 1))
            module.k_proj.weight.data = self.qkv_proj_weight[1, :, :].transpose(0, 1)
            reference_elimination(module.v_proj.weight.data, self.qkv_proj_weight[2, :, :].transpose(0, 1))
            module.v_proj.weight.data = self.qkv_proj_weight[2, :, :].transpose(0, 1)
            if module.q_proj.bias is not None:
                self.qkv_proj_bias = (
                    torch.stack([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias])
                    .contiguous()
                    .view([3, -1])
                )
                reference_elimination(module.q_proj.bias.data, self.qkv_proj_bias[0])
                module.q_proj.bias.data = self.qkv_proj_bias[0]
                reference_elimination(module.k_proj.bias.data, self.qkv_proj_bias[1])
                module.k_proj.bias.data = self.qkv_proj_bias[1]
                reference_elimination(module.v_proj.bias.data, self.qkv_proj_bias[2])
                module.v_proj.bias.data = self.qkv_proj_bias[2]
        else:
            group = self.num_heads // self.num_kv_heads
            q_proj = module.q_proj.weight.view(self.num_kv_heads, group, self.head_dim, self.embed_dim)
            k_proj = module.k_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            v_proj = module.v_proj.weight.view(self.num_kv_heads, 1, self.head_dim, self.embed_dim)
            self.qkv_proj_weight = torch.cat([q_proj, k_proj, v_proj], dim=1).view(
                [self.num_kv_heads, group + 2, self.head_dim, self.embed_dim]
            )
            reference_elimination(
                module.q_proj.data,
                self.qkv_proj_weight[:, :group, :, :].view(
                    [self.num_kv_heads * group * self.head_dim, self.embed_dim]
                ),
            )
            module.q_proj.data = self.qkv_proj_weight[:, :group, :, :].view(
                [self.num_kv_heads * group * self.head_dim, self.embed_dim]
            )
            reference_elimination(
                module.k_proj.data,
                self.qkv_proj_weight[:, group, :, :].view([self.num_kv_heads * self.head_dim, self.embed_dim]),
            )
            module.k_proj.data = self.qkv_proj_weight[:, group, :, :].view(
                [self.num_kv_heads * self.head_dim, self.embed_dim]
            )
            reference_elimination(
                module.v_proj.data,
                self.qkv_proj_weight[:, group + 1, :, :].view([self.num_kv_heads * self.head_dim, self.embed_dim]),
            )
            module.v_proj.data = self.qkv_proj_weight[:, group + 1, :, :].view(
                [self.num_kv_heads * self.head_dim, self.embed_dim]
            )
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.view(self.num_kv_heads, group, self.head_dim)
                k_bias = module.k_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                v_bias = module.v_proj.bias.view(self.num_kv_heads, 1, self.head_dim)
                self.qkv_proj_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(
                    [self.num_kv_heads, group + 2, self.head_dim]
                )
                reference_elimination(module.q_proj.bias.data, self.qkv_proj_bias[:, :group, self.head_dim].view(-1))
                module.q_proj.bias.data = self.qkv_proj_bias[:, :group, self.head_dim].view(-1)
                reference_elimination(module.k_proj.bias.data, self.qkv_proj_bias[:, group, self.head_dim].view(-1))
                module.k_proj.bias.data = self.qkv_proj_bias[:, group, self.head_dim].view(-1)
                reference_elimination(
                    module.v_proj.bias.data, self.qkv_proj_bias[:, group + 1, self.head_dim].view(-1)
                )
                module.v_proj.bias.data = self.qkv_proj_bias[:, group + 1, self.head_dim].view(-1)
        self.o_proj_weight = module.o_proj.weight.transpose(0, 1).contiguous()
        reference_elimination(module.o_proj.weight.data, self.o_proj_weight.transpose(0, 1))
        module.o_proj.weight.data = self.o_proj_weight.transpose(0, 1)
        self.o_proj_bias = module.o_proj.bias
        

class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.distributed = distributed
        self.mlp_impl = None
        if optimized_module is not None:
            self.mlp_impl = optimized_module
        self.port_parameter(module)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        up = torch.ops.torch_ipex.mm_silu(hidden_states, self.gate_proj_weight)
        out = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj_weight, up)
        out = matmul_add_add(out, self.down_proj_weight, self.down_proj_bias, residual)
        return out

    def port_parameter(self, module):
        self.up_proj_weight = module.up_proj.weight.transpose(0, 1).contiguous()
        reference_elimination(module.up_proj.weight.data, self.up_proj_weight.transpose(0, 1))
        module.up_proj.weight.data = self.up_proj_weight.transpose(0, 1)
        self.gate_proj_weight = module.gate_proj.weight.transpose(0, 1).contiguous()
        reference_elimination(module.gate_proj.weight.data, self.gate_proj_weight.transpose(0, 1))
        module.gate_proj.weight.data = self.gate_proj_weight.transpose(0, 1)
        self.down_proj_weight = module.down_proj.weight.transpose(0, 1).contiguous()
        reference_elimination(module.down_proj.weight.data, self.down_proj_weight.transpose(0, 1))
        module.down_proj.weight.data = self.down_proj_weight.transpose(0, 1)
        self.up_proj_bias = module.up_proj.bias
        self.gate_proj_bias = module.gate_proj.bias
        self.down_proj_bias = module.down_proj.bias
        
        
class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, config, distributed=False) -> None:
        super().__init__()
        self.layer_idx = module.self_attn.layer_idx
        self.attn = _IPEXLlamaAttention(module.self_attn, config, distributed)
        self.mlp = _IPEXLlamaMLP(module.mlp, config, distributed)
        self.input_layernorm = ipex.llm.modules.RMSNorm(
            module.input_layernorm.weight, module.input_layernorm.variance_epsilon
        )
        self.post_attention_layernorm = ipex.llm.modules.RMSNorm(
            module.post_attention_layernorm.weight, module.post_attention_layernorm.variance_epsilon
        )

    def preprocess_for_optimize(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        postion_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attention: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        return hidden_states, attention_mask, postion_ids, past_key_value


    def postprocess_for_optimize(
        self, hidden_states, output_attention, use_cache, self_attn_weight, present_key_value, **kwargs
    ):
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        if output_attention:
            outputs += (self_attn_weight,)

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        outputs = self.preprocess_for_optimize(
            hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs
        )
        (hidden_states, attention_mask, position_ids, past_key_value) = outputs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, self_attn_weight = self.attn(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            None,
            residual,
            **kwargs,
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual, **kwargs)

        outputs = self.postprocess_for_optimize(
            hidden_states, output_attentions, use_cache, self_attn_weight, present_key_value, **kwargs
        )

        return outputs