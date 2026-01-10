import os
import torch

# Disable torch compilation for ROCm compatibility
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

from lmdeploy import pipeline, TurbomindEngineConfig, PytorchEngineConfig, GenerationConfig
from .base import BaseModel


class LMDeployModel(BaseModel):
    def __init__(self,
            device='cuda',
            cache_size_mb=100,
            **kwargs):
        assert device == 'cuda', "lmdeploy only supports cuda devices, consider changing device or using a different backend instead."
        cache_size_ratio = cache_size_mb * 1024**2 / torch.cuda.get_device_properties('cuda').total_memory

        # Detect if we're on ROCm (AMD GPU)
        is_rocm = 'Radeon' in torch.cuda.get_device_name(0) or 'AMD' in torch.cuda.get_device_name(0)
        self.is_rocm = is_rocm
        self.captured_hidden_states = []

        if is_rocm:
            # Use PytorchEngine for ROCm with compilation disabled
            # Disable CUDA graphs so hooks work during generation
            backend_config = PytorchEngineConfig(
                cache_max_entry_count=cache_size_ratio,
                enable_prefix_caching=False,
                eager_mode=True  # Disable CUDA graphs to allow hooks during generation
            )
        else:
            # Use TurbomindEngine for CUDA
            backend_config = TurbomindEngineConfig(cache_max_entry_count=cache_size_ratio)

        self.pipeline = pipeline('ekwek/Soprano-80M',
            log_level='ERROR',
            backend_config=backend_config)

        # For ROCm, we need to hook into the model to capture hidden states
        if is_rocm:
            self._setup_hidden_state_capture()

    def _setup_hidden_state_capture(self):
        """Setup hooks to capture hidden states from the model on ROCm"""
        # Access the underlying model from the pipeline
        try:
            # Get the model from the pytorch engine executor
            # Path: pipeline.engine.executor.model_agent.patched_model.model.model.norm
            model = self.pipeline.engine.executor.model_agent.patched_model.model

            # Hook into model.norm to capture hidden states before lm_head
            def hook_fn(module, input, output):
                # Store the hidden states (output from final norm layer)
                # The norm layer may return a tuple (hidden_states, residual)
                # Extract the actual hidden states tensor
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # During autoregressive generation, we get one call per generated token
                # We only want the last token's hidden state from each forward pass
                # Keep on GPU - no need to transfer to CPU since we use it immediately on GPU
                if hidden_states.dim() == 3:
                    # (batch, seq_len, hidden_dim) - take last position
                    # Detach to avoid gradient tracking, keep on GPU
                    self.captured_hidden_states.append(hidden_states[:, -1:, :].detach())
                else:
                    self.captured_hidden_states.append(hidden_states.detach())

            # Register hook on the final normalization layer
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                model.model.norm.register_forward_hook(hook_fn)
                print("✓ Hidden state capture hook registered on model.norm")
            else:
                print("Warning: Could not find model.norm layer for hidden state capture")
        except Exception as e:
            print(f"Warning: Could not setup hidden state capture: {e}")

    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        # Clear previously captured hidden states
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
            # On ROCm, use captured hidden states instead of response.last_hidden_state
            if self.is_rocm and len(self.captured_hidden_states) > 0:
                # During generation, the hook is called once per forward pass
                # For batch generation, we need to extract hidden states for this specific prompt
                # Concatenate all captured states and extract for this batch item
                try:
                    # Debug: print capture info
                    # print(f"DEBUG: Captured {len(self.captured_hidden_states)} forward passes")
                    # if len(self.captured_hidden_states) > 0:
                    #     print(f"DEBUG: First capture shape: {self.captured_hidden_states[0].shape}")

                    # Stack all captures: each is (batch, 1, hidden_dim)
                    # All tensors are already on GPU, no need to move
                    stacked = torch.cat(self.captured_hidden_states, dim=1)  # (batch, total_tokens, hidden_dim)

                    # Extract for this batch item
                    if i < stacked.size(0):
                        hidden_state = stacked[i]  # (total_tokens, hidden_dim)

                        # Skip prompt tokens, keep only generated
                        num_input = response.input_token_len
                        num_generated = response.generate_token_len

                        # Debug
                        # print(f"DEBUG: Total captured tokens: {hidden_state.size(0)}, Input tokens: {num_input}, Generated: {num_generated}")

                        # The captured states include both prompt processing and generation
                        # We want only the generation part
                        if hidden_state.size(0) > num_generated:
                            # Take last num_generated tokens
                            hidden_state = hidden_state[-num_generated:, :]

                        # hidden_state already on GPU, no need to move
                    else:
                        hidden_state = None
                except Exception as e:
                    print(f"Warning: Failed to extract hidden states: {e}")
                    import traceback
                    traceback.print_exc()
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
        # Clear previously captured hidden states
        if self.is_rocm:
            self.captured_hidden_states = []
            token_count = 0

        gen_config=GenerationConfig(output_last_hidden_state='generation',
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=512)
        responses = self.pipeline.stream_infer([prompt], gen_config=gen_config)

        for response in responses:
            # On ROCm, use captured hidden states (already on GPU)
            if self.is_rocm and len(self.captured_hidden_states) > token_count:
                hidden_state = self.captured_hidden_states[token_count]
                token_count += 1
            else:
                hidden_state = response.last_hidden_state

            yield {
                'finish_reason': response.finish_reason,
                'hidden_state': hidden_state
            }
