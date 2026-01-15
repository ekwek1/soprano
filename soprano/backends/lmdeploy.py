import os
import torch
from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig
from .base import BaseModel


class LMDeployModel(BaseModel):
    def __init__(self,
            device='cuda',
            cache_size_mb=100,
            model_path=None,
            **kwargs):
        assert device == 'cuda', "lmdeploy only supports cuda devices, consider changing device or using a different backend instead."
        
        # Calculate cache size
        try:
            total_mem = torch.cuda.get_device_properties('cuda').total_memory
            cache_size_ratio = cache_size_mb * 1024**2 / total_mem
        except:
            cache_size_ratio = 0.01  # Fallback
            
        # Detect ROCm (AMD GPU)
        self.is_rocm = False
        try:
            dev_name = torch.cuda.get_device_name(0)
            if 'Radeon' in dev_name or 'AMD' in dev_name:
                self.is_rocm = True
        except:
            pass
            
        self.captured_hidden_states = []
        
        pipeline_kwargs = {}

        if self.is_rocm:
            print("Using PytorchEngine for ROCm")
            # ROCm/AMD specific config
            backend_config = PytorchEngineConfig(
                cache_max_entry_count=cache_size_ratio,
                enable_prefix_caching=False,
                eager_mode=True
            )
            
            # Use bfloat16 if supported (now backed by fixed library check)
            if torch.cuda.is_bf16_supported():
                print("Enabling bfloat16 support (ROCm)")
                pipeline_kwargs['dtype'] = 'bfloat16'
        else:
            # Standard Turbomind for NVIDIA
            backend_config = TurbomindEngineConfig(cache_max_entry_count=cache_size_ratio)
        
        # Use local model if path provided, otherwise use HuggingFace
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'
        
        self.pipeline = pipeline(model_name_or_path,
            log_level='ERROR',
            backend_config=backend_config,
            **pipeline_kwargs)

        # Register hooks for ROCm
        if self.is_rocm:
            self._setup_hidden_state_capture()

    def _setup_hidden_state_capture(self):
        """Attaches hooks to capture hidden states on AMD cards"""
        try:
            # Access internal model structure
            # Path depends on lmdeploy version/backend structure
            # For PytorchEngine: engine.executor.model_agent.patched_model.model
            model = self.pipeline.engine.executor.model_agent.patched_model.model

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Store detached tensor
                # We normalize to float32 immediately to ensure decoder compatibility and avoid noise
                if hidden_states.dim() == 3:
                    self.captured_hidden_states.append(hidden_states[:, -1:, :].detach().float())
                else:
                    self.captured_hidden_states.append(hidden_states.detach().float())

            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                model.model.norm.register_forward_hook(hook_fn)
                print("✓ ROCm Hook Registered successfully (bfloat16 capable)")
            else:
                print("Warning: Could not find model.norm for ROCm hook")
        except Exception as e:
            print(f"Warning: Failed to setup ROCm hooks: {e}")

    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        
        # Reset buffer for ROCm
        if self.is_rocm:
            self.captured_hidden_states = []

        gen_config=GenerationConfig(output_last_hidden_state='generation',
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=512)
        responses = self.pipeline(prompts, gen_config=gen_config)
        res = []
        for i, response in enumerate(responses):
            # ROCm logic: reconstruct from hooks
            if self.is_rocm and len(self.captured_hidden_states) > 0:
                try:
                    stacked = torch.cat(self.captured_hidden_states, dim=1)
                    if i < stacked.size(0): 
                        # Precise alignment logic
                        hidden_state = stacked[i]
                        num_gen = response.generate_token_len
                        
                        # We need the hidden state that PRODUCED the token.
                        # Index 0 (Prefill end) -> Produces Token 0
                        # ...
                        # Index N-1 (Gen N-1) -> Produces Token N (EOS)
                        # Index N (Gen N/EOS) -> Unused/Garbage
                        
                        if hidden_state.size(0) >= num_gen:
                            # Take first num_gen states (0 to N-1)
                            hidden_state = hidden_state[:num_gen, :]
                        else:
                             # Fallback if weirdly short (unlikely)
                             print(f"Warning: captured states too short {hidden_state.size(0)} < {num_gen}")
                    else:
                        hidden_state = stacked[0][:response.generate_token_len, :] if self.captured_hidden_states else None
                except Exception as e:
                    print(f"ROCm Capture Error: {e}")
                    hidden_state = None
            else:
                hidden_state = response.last_hidden_state
            
            res.append({
                'finish_reason': response.finish_reason,
                'hidden_state': hidden_state
            })
        return res

    def stream_infer(self,
            prompt,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        
        if self.is_rocm:
            self.captured_hidden_states = []
            
        gen_config=GenerationConfig(output_last_hidden_state='generation',
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=512)
        responses = self.pipeline.stream_infer([prompt], gen_config=gen_config)
        
        yielded_token_count = 0
        
        for response in responses:
            if self.is_rocm:
                all_tokens = response.token_ids
                num_generated = len(all_tokens)
                
                if num_generated > yielded_token_count:
                    # Calculate offset: verify we have enough captured states
                    # The hook captures everything, including prompt processing. 
                    # We align from the END of the capture buffer.
                    
                    total_captured = len(self.captured_hidden_states)
                    
                    # Ensure we don't index past what we have
                    # We want the index corresponding to the newly generated token
                    # If we generated N tokens total, we want the Nth from last captured state?
                    # No, captured states accrue sequentially.
                    # Prompt processing might add M items. Generation adds 1 item per step.
                    # So index M + (num_generated - 1) is current token.
                    
                    # A robust way is to assume strict 1:1 mapping for generated tokens at the END of the list.
                    # The last state captured corresponds to the last token generated.
                    # The (last - 1) state corresponds to (last - 1) token.
                    
                    new_tokens_count = num_generated - yielded_token_count
                    
                    for k in range(new_tokens_count):
                        # We want the token at 'yielded_token_count + k' (0-indexed relative to gen)
                        # Captured Index 0 corresponds to Gen Token 0.
                        
                        idx_in_gen = yielded_token_count + k
                        # Direct mapping:
                        idx_in_capture = idx_in_gen
                        
                        if 0 <= idx_in_capture < total_captured:
                            hidden_state = self.captured_hidden_states[idx_in_capture]
                            
                            if hidden_state.dim() == 3 and hidden_state.shape[1] == 1:
                                hidden_state = hidden_state.squeeze(1)

                            yield {
                                'finish_reason': None,
                                'hidden_state': hidden_state
                            }
                    
                    yielded_token_count = num_generated

                if response.finish_reason:
                     yield {
                        'finish_reason': response.finish_reason,
                        'hidden_state': self.captured_hidden_states[-1].squeeze(1) if len(self.captured_hidden_states)>0 else None
                    }
            else:
                yield {
                    'finish_reason': response.finish_reason,
                    'hidden_state': response.last_hidden_state
                }