from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput, BaseModelOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging
import torch.nn.functional as F
import numpy as np


from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice import VibeVoiceConfig


logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

def _vv_is_array_like(x):
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, np.ndarray):
        return True
    return False

def _vv_peel(x):
    # Peel [batch][stream]/nested lists until we hit tensor/ndarray
    while isinstance(x, (list, tuple)) and len(x) > 0 and not _vv_is_array_like(x):
        x = x[0]
    return x

def _vv_extract_frames(enc_out):
    """
    Supports both new (object) and old (list/tuple) encoder outputs.
    Returns an array/tensor of frames/tokens for 1 batch.
    """
    candidate_attrs = [
        "frames", "frame_ids",
        "codes", "tokens", "token_ids", "ids",
        "acoustic_frames", "acoustic_codes", "acoustic_tokens", "acoustic_ids",
        "semantic_frames", "semantic_codes", "semantic_tokens", "semantic_ids",
        "values", "data",
    ]
    for name in candidate_attrs:
        if hasattr(enc_out, name):
            val = getattr(enc_out, name)
            val = _vv_peel(val)
            if _vv_is_array_like(val):
                return val

    # Mapping protocol (e.g., dict-like)
    try:
        items = dict(enc_out)
        for _, v in items.items():
            v = _vv_peel(v)
            if _vv_is_array_like(v):
                return v
    except Exception:
        pass

    # Iterable fallback
    try:
        it = list(enc_out)
        v = _vv_peel(it)
        if _vv_is_array_like(v):
            return v
    except Exception:
        pass

    # __dict__ fallback
    if hasattr(enc_out, "__dict__"):
        for v in enc_out.__dict__.values():
            v = _vv_peel(v)
            if _vv_is_array_like(v):
                return v

    # Old API: nested list/tuple [batch][stream]
    if isinstance(enc_out, (list, tuple)):
        return enc_out[0][0]

    raise TypeError(f"Ukjent encoder output-type: {type(enc_out)}")

def _vv_resize_time(feat: torch.Tensor, target_len: int) -> torch.Tensor:
    # feat: [T, D]
    if feat.dim() != 2:
        raise ValueError(f"expected [T, D], got {feat.shape}")
    T, D = feat.shape
    if T == target_len:
        return feat
    x = feat.transpose(0, 1).unsqueeze(0)  # [1, D, T]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.squeeze(0).transpose(0, 1).contiguous()  # [target_len, D]
    return x

def _vv_ensure_4d(x: torch.Tensor) -> torch.Tensor:
    # Ensure [B, 1, L, H]
    if x.dim() == 4:
        return x
    if x.dim() == 3:
        # Treat [B, L, H]
        return x.unsqueeze(1) if x.size(-1) > 1 else x.unsqueeze(-1)
    if x.dim() == 2:
        B, L = x.shape
        return x.view(B, 1, L, 1)
    raise ValueError(f"unexpected x.dim={x.dim()} shape={x.shape}")

def _vv_time_mask(mask: Optional[torch.Tensor], L: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Return [B,1,L] mask from various shapes:
    - [B,L] -> [B,1,L]
    - [B,1,L] -> [B,1,L]
    - [B,L,H] or [B,1,L,H] -> any over H -> [B,1,L]
    """
    if mask is None:
        return None
    m = mask
    if m.dim() == 4:
        return m.any(dim=-1)
    if m.dim() == 3:
        return m if m.size(1) == 1 else m.any(dim=-1, keepdim=True)
    if m.dim() == 2:
        return m.unsqueeze(1)
    raise ValueError(f"unexpected mask.dim={m.dim()} shape={m.shape}")

def _vv_to_2d_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    m = mask
    if m.dim() == 4:
        m = m.any(dim=-1).squeeze(1)
    elif m.dim() == 3:
        m = m.squeeze(1) if m.size(1) == 1 else m.any(dim=-1)
    elif m.dim() == 2:
        pass
    else:
        raise ValueError(f"unexpected attention_mask dim={m.dim()} shape={m.shape}")
    return m.to(dtype=torch.long)
@dataclass
class VibeVoiceCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    diffusion_loss: Optional[torch.FloatTensor] = None
    speech_token_num: Optional[int] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):    
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


# @auto_docstring
class VibeVoicePreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, VibeVoiceDiffusionHead):
            module.initialize_weights()
            return

        # Use the language model's initializer_range if available
        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value
            
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

# @auto_docstring
class VibeVoiceModel(VibeVoicePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        if hasattr(config, 'torch_dtype') and config.torch_dtype is not None:
            if isinstance(config.torch_dtype, str):
                dtype = getattr(torch, config.torch_dtype)
            else:
                dtype = config.torch_dtype
        else:
            dtype = torch.float32
        
        # Initialize Qwen2 model for language modeling
        lm_config = config.decoder_config 
        self.language_model = AutoModel.from_config(lm_config)
        
        # Initialize speech components if needed
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).to(dtype)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config).to(dtype)

        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, lm_config.hidden_size).to(dtype)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size).to(dtype)
        
        # Register scaling factors as buffers - use 1D tensors for FSDP compatibility
        self.register_buffer('speech_scaling_factor', torch.tensor(float('nan')))  
        self.register_buffer('speech_bias_factor', torch.tensor(float('nan')))

        # Initialize prediction head for speech generation
        self.prediction_head = AutoModel.from_config(config.diffusion_head_config).to(dtype)

        # Initialize noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )
    
    def get_input_embeddings(self):
        if hasattr(self.language_model, 'embed_tokens'):
            # If the language model has an embed_tokens attribute, return it
            return self.language_model.embed_tokens
        
        for name, attr in self.language_model.fullmap.items(): # parallel by nnscaler, the name is changed
            if attr.orig_name == 'embed_tokens.weight':
                return getattr(self.language_model, name)
        assert False, 'should not arrive here'

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()
            
        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        if not return_dict:
            return outputs
            
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VibeVoiceForConditionalGeneration(VibeVoicePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceModel(config)
        self.vocab_size = config.decoder_config.vocab_size
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.language_model = decoder

    def get_decoder(self):
        return self.model.language_model

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if getattr(self.config.decoder_config, 'tie_word_embeddings', False):
            # The standard PreTrainedModel method will handle the tying.
            # It typically does a simple parameter object assignment, which is
            # CORRECT to do BEFORE FSDP wraps the model.
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if hasattr(input_embeddings, 'weight'):
                output_embeddings.weight = input_embeddings.weight
            else:
                # maybe returned input_embeddings a tensor directly
                output_embeddings.weight = input_embeddings

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                    "constant",
                    0,
                )
            print("✅ Tied input and output embeddings using standard assignment.")
        else:
            print("ℹ️  tie_word_embeddings is False, not tying weights.")

    # Also, ensure set_output_embeddings is safe, though your implementation looks okay.
    # The key is to avoid calling it after accelerator.prepare().
    def set_output_embeddings(self, new_embeddings):
        # Your current implementation using data.copy_ is good practice,
        # but the best way is to not call this after prepare().
        self.lm_head = new_embeddings

    def forward_speech_features(
            self, 
            speech_tensors=None, 
            speech_masks=None, 
            speech_type="audio", 
            return_unmask=False
        ):
        if speech_tensors is None:
            # Use config to get vae_dim instead of non-existent self.args
            vae_dim = self.config.acoustic_tokenizer_config.vae_dim
            audio_features = torch.zeros(1, 1, vae_dim).to(self.get_input_embeddings().weight)
            connect_features = self.model.acoustic_connector(audio_features)
            return audio_features, connect_features
        else:
            with torch.no_grad():
                if speech_type == "audio":
                    with torch.no_grad():
                        enc_out = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                        frames = None
                        if hasattr(enc_out, "sample") and callable(enc_out.sample):
                            try:
                                frames = enc_out.sample(self.model.acoustic_tokenizer.std_dist_type)
                            except Exception:
                                try:
                                    frames = enc_out.sample()
                                except Exception:
                                    frames = None
                        if frames is None:
                            frames = _vv_extract_frames(enc_out)
                        frames = _vv_peel(frames)
                        if not torch.is_tensor(frames):
                            frames = torch.as_tensor(frames, device=speech_tensors.device)
                        if frames.dim() == 1:
                            frames = frames.unsqueeze(0)
                        if frames.dim() == 2:
                            frames = frames.unsqueeze(0)
                        if not hasattr(self, "_vv_dbg_once"):
                            self._vv_dbg_once = True
                            try:
                                print("ENC OUT attrs:", [a for a in dir(enc_out) if not a.startswith("_")])
                                print("FRAMES shape:", getattr(frames, "shape", None), "dtype:", getattr(frames, "dtype", None))
                            except Exception:
                                pass
                    audio_tokens = frames

                elif speech_type == "vae":
                    # Use config to get vae_dim instead of non-existent self.args
                    vae_dim = self.config.acoustic_tokenizer_config.vae_dim
                    speech_mode = speech_tensors.reshape(speech_tensors.size(0), -1, vae_dim)

                    # gaussian sample from the speech_mode
                    batch_size = speech_mode.size(0)
                    value = self.model.acoustic_tokenizer.fix_std / 0.8
                    std = torch.randn(batch_size, dtype=speech_mode.dtype, device=speech_mode.device) * value
                    std = std.view(-1, *[1] * (speech_mode.dim() - 1))
                    audio_tokens = speech_mode + std * torch.randn(speech_mode.shape).to(speech_mode)
                else:
                    raise NotImplementedError(f"Speech type {speech_type} not implemented")
                
                if torch.isnan(self.model.speech_scaling_factor) or torch.isnan(self.model.speech_bias_factor):
                    scaling_factor = 1. / audio_tokens[speech_masks].flatten().std()
                    bias_factor = -audio_tokens[speech_masks].flatten().mean()
                    
                    # Only use distributed operations if the process group is initialized
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(scaling_factor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(bias_factor, op=dist.ReduceOp.SUM)
                        world_size = dist.get_world_size()
                        self.model.speech_scaling_factor.copy_(scaling_factor / world_size)  
                        self.model.speech_bias_factor.copy_(bias_factor / world_size)
                        print(f"Speech scaling factor (distributed): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
                    else:
                        # Single process case
                        self.model.speech_scaling_factor.copy_(scaling_factor)  
                        self.model.speech_bias_factor.copy_(bias_factor)
                        print(f"Speech scaling factor (single process): {self.model.speech_scaling_factor}, bias factor: {self.model.speech_bias_factor}", flush=True)
                    
                audio_features = (audio_tokens + self.model.speech_bias_factor) * self.model.speech_scaling_factor
            
            connect_features = self.model.acoustic_connector(audio_features)
            if return_unmask:
                return audio_features, connect_features
            return audio_features[speech_masks], connect_features[speech_masks]
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # New arguments for speech processing and loss calculation
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speeches_loss_input: Optional[torch.FloatTensor] = None,
        speech_semantic_tensors: Optional[torch.FloatTensor] = None, 
        acoustic_input_mask: Optional[torch.BoolTensor] = None,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        ddpm_batch_mul: int = 1,
        **kwargs: Optional[Dict[str, Union[torch.Tensor, str]]],
        ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        x = self.get_input_embeddings()(input_ids)

        # Fail-safe: if semantic features are missing, fabricate safe zeros matching connector input
        try:
            connector_device = next(self.model.semantic_connector.parameters()).device
            connector_dtype = next(self.model.semantic_connector.parameters()).dtype
            connector_in_features = self.model.semantic_connector.fc1.in_features
        except Exception:
            connector_device = x.device
            connector_dtype = x.dtype
            connector_in_features = x.size(-1)

        if speech_semantic_tensors is None:
            batch_size = input_ids.size(0) if input_ids is not None else 1
            speech_semantic_tensors = torch.zeros(
                (batch_size, connector_in_features), device=connector_device, dtype=connector_dtype
            )

        semantic_speech_all_connect_features = self.model.semantic_connector(speech_semantic_tensors)
        if speeches_loss_input is not None:
            # only part audio need diffuse
            speech_all_features, speech_all_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                    return_unmask=True
                )
            if speech_tensors is not None:
                # Insert features along time using a 1D time mask, resizing if needed
                x = _vv_ensure_4d(x)
                B, _, L, H = x.shape
                if acoustic_input_mask is not None:
                    acoustic_input_mask = _vv_ensure_4d(acoustic_input_mask)
                    time_mask = acoustic_input_mask.any(dim=-1)  # [B, 1, L]
                else:
                    # Derive time mask from speech_masks if available, otherwise use all True
                    if speech_masks is not None:
                        sm = speech_masks
                        if sm.dim() == 1:
                            sm = sm.unsqueeze(0).unsqueeze(0)
                        elif sm.dim() == 2:
                            sm = sm.unsqueeze(1)
                        elif sm.dim() == 3 and sm.size(1) != 1:
                            sm = sm.any(dim=1, keepdim=True)
                        elif sm.dim() == 4:
                            sm = sm.any(dim=-1)
                        time_mask = sm.bool()
                    else:
                        time_mask = torch.ones((B, 1, L), dtype=torch.bool, device=x.device)
                target_len = int(time_mask[0, 0].sum().item())
                idx = time_mask[0, 0].nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    insert_feat = speech_all_connect_features[speech_masks]  # [T, H]
                    if insert_feat.size(0) != target_len:
                        insert_feat = _vv_resize_time(insert_feat, target_len)
                    insert_feat = insert_feat.to(device=x.device, dtype=x.dtype).contiguous()
                    x = x.clone()
                    if semantic_speech_all_connect_features is not None:
                        sem_feat = semantic_speech_all_connect_features[speech_masks]
                        if sem_feat.size(0) != target_len:
                            sem_feat = _vv_resize_time(sem_feat, target_len)
                        sem_feat = sem_feat.to(device=x.device, dtype=x.dtype).contiguous()
                        x[0, 0, idx, :] = (insert_feat + sem_feat)
                    else:
                        x[0, 0, idx, :] = insert_feat
                speech_features = speech_all_features[speeches_loss_input.unsqueeze(-1) & speech_masks] # only part audio need diffuse
                speech_connect_features = speech_all_connect_features[speeches_loss_input.unsqueeze(-1) & speech_masks]
        else:
            speech_features, speech_connect_features = self.forward_speech_features(
                    speech_tensors=speech_tensors.type_as(x) if speech_tensors is not None else None,
                    speech_masks=speech_masks,
                    speech_type=kwargs.get("speech_type", "audio"),
                )
            if speech_tensors is not None:
                # Insert features along time using a 1D time mask, resizing if needed
                x = _vv_ensure_4d(x)
                B, _, L, H = x.shape
                if acoustic_input_mask is not None:
                    acoustic_input_mask = _vv_ensure_4d(acoustic_input_mask)
                    time_mask = acoustic_input_mask.any(dim=-1)  # [B, 1, L]
                else:
                    if speech_masks is not None:
                        sm = speech_masks
                        if sm.dim() == 1:
                            sm = sm.unsqueeze(0).unsqueeze(0)
                        elif sm.dim() == 2:
                            sm = sm.unsqueeze(1)
                        elif sm.dim() == 3 and sm.size(1) != 1:
                            sm = sm.any(dim=1, keepdim=True)
                        elif sm.dim() == 4:
                            sm = sm.any(dim=-1)
                        time_mask = sm.bool()
                    else:
                        time_mask = torch.ones((B, 1, L), dtype=torch.bool, device=x.device)
                target_len = int(time_mask[0, 0].sum().item())
                idx = time_mask[0, 0].nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    insert_feat = speech_connect_features  # [T, H]
                    if insert_feat.size(0) != target_len:
                        insert_feat = _vv_resize_time(insert_feat, target_len)
                    insert_feat = insert_feat.to(device=x.device, dtype=x.dtype).contiguous()
                    x = x.clone()
                    x[0, 0, idx, :] = insert_feat

        # Back to 3D for LLM and normalize attention mask
        x = x.squeeze(1)  # [B, L, H]
        attention_mask = _vv_to_2d_mask(attention_mask)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=x,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            # The custom CE loss with masking is calculated in the training script.
            # We leave the standard loss calculation here as None.
            pass

        # --- Diffusion Loss Calculation ---
        diffusion_loss = None
        # This block is executed only if we are in a context that involves speech.
        if (speech_tensors is not None 
            and acoustic_loss_mask is not None 
            and acoustic_loss_mask.any()):
            condition_features = hidden_states[acoustic_loss_mask]
            
            speech_len, latent_size = speech_features.shape
            
            noise = torch.randn(
                (speech_len * ddpm_batch_mul, latent_size),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            timesteps = torch.multinomial(
                torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                speech_len * ddpm_batch_mul,
                replacement=True,
            ).to(hidden_states.device)

            speech_features_repeated = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
            condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)

            noisy_speech_features = self.model.noise_scheduler.add_noise(
                speech_features_repeated, noise, timesteps
            )
            
            model_output = self.model.prediction_head(
                noisy_speech_features, 
                timesteps.type_as(x), 
                condition_features_repeated
            )

            prediction_type = self.config.diffusion_head_config.prediction_type
            if prediction_type == "epsilon":
                target_for_loss = noise
            elif prediction_type == "v_prediction":
                target_for_loss = self.model.noise_scheduler.get_velocity(
                    speech_features_repeated, noise, timesteps
                )
            else:
                raise NotImplementedError(f"Prediction type {prediction_type} not implemented")

            diffusion_loss = F.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')
            if latent_size > 0 and ddpm_batch_mul > 0:
                diffusion_loss = diffusion_loss / latent_size / ddpm_batch_mul
            else:
                diffusion_loss = torch.tensor(0.0, device=diffusion_loss.device)
        
        else:
            # Dummy loss for DDP to work when there are no speech samples in a batch,
            # but we are in a speech context.
            diffusion_loss = sum(p.sum() for p in self.model.prediction_head.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.model.acoustic_connector.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.model.semantic_connector.parameters()) * 0.0
        # --- End Diffusion Loss Calculation ---

        if not return_dict:
            output = (logits, speech_len) + outputs.to_tuple()[1:]
            return (loss, diffusion_loss) + output

        return VibeVoiceCausalLMOutputWithPast(
            loss=loss,
            diffusion_loss=diffusion_loss,
            speech_token_num=speech_len if speech_tensors is not None else 0,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoModel.register(VibeVoiceConfig, VibeVoiceModel)
AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

__all__ = [
    "VibeVoiceModel",
    "VibeVoicePreTrainedModel",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceCausalLMOutputWithPast",
    "VibeVoiceGenerationOutput",
]