import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


def _detect_rocm():
    """Detect if running on ROCm (AMD GPU)"""
    try:
        dev_name = torch.cuda.get_device_name(0)
        return 'Radeon' in dev_name or 'AMD' in dev_name or 'gfx' in dev_name.lower()
    except:
        return False


def _setup_rocm_env():
    """Setup environment variables for optimal ROCm performance"""
    # Use hipBLASLt for better performance on RDNA3/CDNA
    os.environ.setdefault('TORCH_BLAS_PREFER_HIPBLASLT', '1')
    # Triton attention is best for ROCm
    os.environ.setdefault('VLLM_USE_TRITON_FLASH_ATTN', '1')
    # Disable NCCL P2P for single GPU (avoids warnings)
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')


class VLLMModel(BaseModel):
    """
    vLLM backend for Soprano TTS optimized for ROCm.

    Uses hybrid approach:
    - vLLM for fast token generation (with ROCm optimizations)
    - Transformers for hidden state extraction (compiled for speed)
    """

    def __init__(self,
            device='cuda',
            model_path=None,
            gpu_memory_utilization=0.4,
            **kwargs):
        assert device == 'cuda', "vLLM only supports CUDA devices (including ROCm)"

        self.device = device
        self.is_rocm = _detect_rocm()

        if self.is_rocm:
            _setup_rocm_env()
            print(f"Detected AMD GPU: {torch.cuda.get_device_name(0)}")

        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'

        # bfloat16 is well supported on RDNA3 and better for quality
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        dtype_str = 'bfloat16' if self.dtype == torch.bfloat16 else 'float16'

        print(f"Initializing vLLM backend: dtype={dtype_str}, ROCm={self.is_rocm}")

        # vLLM configuration optimized for ROCm
        vllm_kwargs = {
            'model': model_name_or_path,
            'dtype': dtype_str,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': 1024,  # Soprano doesn't need long context
            'disable_log_stats': True,
        }

        if self.is_rocm:
            # ROCm optimizations
            vllm_kwargs.update({
                'enforce_eager': False,  # Allow torch.compile on ROCm
                'disable_custom_all_reduce': True,  # Single GPU
            })
        else:
            vllm_kwargs['enforce_eager'] = True

        self.llm = LLM(**vllm_kwargs)

        # Transformers model for hidden state extraction
        print("Loading transformers model for hidden state extraction...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            device_map=device,
            attn_implementation='sdpa',  # Use PyTorch SDPA (works on ROCm)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.hf_model.eval()

        # Note: torch.compile doesn't work well with output_hidden_states=True
        # Keep model uncompiled for hidden state extraction

        self._eos_token_id = self.hf_model.config.eos_token_id
        print("✓ Hybrid vLLM + Transformers backend ready")

    @torch.inference_mode()
    def _get_hidden_states(self, input_ids, generated_ids):
        """Extract hidden states for generated tokens using transformers model"""
        full_ids = torch.cat([input_ids, generated_ids], dim=1)

        outputs = self.hf_model(
            full_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        if outputs.hidden_states is None:
            return None

        # Get last layer hidden states for generated tokens only
        last_hidden = outputs.hidden_states[-1]
        gen_start = input_ids.size(1)
        hidden_states = last_hidden[:, gen_start:, :]

        # Exclude EOS token hidden states
        mask = generated_ids[0] != self._eos_token_id
        hidden_states = hidden_states[:, mask, :]

        if hidden_states.size(1) == 0:
            return None

        return hidden_states

    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):

        sampling_params = SamplingParams(
            top_p=top_p,
            temperature=max(temperature, 0.01),
            repetition_penalty=repetition_penalty,
            max_tokens=512,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        res = []
        for i, output in enumerate(outputs):
            finish_reason = 'stop' if output.outputs[0].finish_reason == 'stop' else 'length'
            generated_token_ids = output.outputs[0].token_ids

            try:
                input_ids = self.tokenizer(
                    prompts[i],
                    return_tensors='pt',
                    add_special_tokens=False
                ).input_ids.to(self.device)

                gen_ids = torch.tensor([generated_token_ids], device=self.device)
                hidden_states = self._get_hidden_states(input_ids, gen_ids)
                hidden_state = hidden_states[0].float()

            except Exception as e:
                print(f"Hidden state extraction error: {e}")
                hidden_state = None

            res.append({
                'finish_reason': finish_reason,
                'hidden_state': hidden_state
            })

        return res

    def stream_infer(self,
            prompt,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        """
        Streaming inference.
        Generates all tokens first, then yields hidden states.
        """
        sampling_params = SamplingParams(
            top_p=top_p,
            temperature=max(temperature, 0.01),
            repetition_penalty=repetition_penalty,
            max_tokens=512,
        )

        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
        output = outputs[0]

        finish_reason = 'stop' if output.outputs[0].finish_reason == 'stop' else 'length'
        generated_token_ids = list(output.outputs[0].token_ids)

        # Remove EOS token
        if generated_token_ids and generated_token_ids[-1] == self._eos_token_id:
            generated_token_ids = generated_token_ids[:-1]

        if not generated_token_ids:
            yield {'finish_reason': finish_reason, 'hidden_state': None}
            return

        try:
            input_ids = self.tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=False
            ).input_ids.to(self.device)

            gen_ids = torch.tensor([generated_token_ids], device=self.device)
            hidden_states = self._get_hidden_states(input_ids, gen_ids)

            if hidden_states is None:
                yield {'finish_reason': finish_reason, 'hidden_state': None}
                return

            num_tokens = hidden_states.size(1)
            for i in range(num_tokens):
                is_last = (i == num_tokens - 1)
                yield {
                    'finish_reason': finish_reason if is_last else None,
                    'hidden_state': hidden_states[0, i:i+1, :].float()
                }

        except Exception as e:
            print(f"Streaming hidden state error: {e}")
            yield {'finish_reason': finish_reason, 'hidden_state': None}
