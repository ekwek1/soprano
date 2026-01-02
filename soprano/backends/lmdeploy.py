import torch
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from .base import BaseModel


class LMDeployModel(BaseModel):
    def __init__(self,
            device='cuda',
            cache_size_mb=100,
            **kwargs):
        # LMDeploy supports both CUDA and CPU
        self.device = device
        
        if device == 'cuda':
            # Original CUDA implementation with cache size optimization
            cache_size_ratio = cache_size_mb * 1024**2 / torch.cuda.get_device_properties('cuda').total_memory
            backend_config = TurbomindEngineConfig(cache_max_entry_count=cache_size_ratio)
            self.pipeline = pipeline('ekwek/Soprano-80M',
                log_level='ERROR',
                backend_config=backend_config)
        elif device == 'cpu':
            # CPU implementation - TurbomindEngineConfig not needed
            # LMDeploy will automatically use CPU inference
            self.pipeline = pipeline('ekwek/Soprano-80M',
                log_level='ERROR')
        else:
            raise ValueError(f"Unsupported device: {device}. Must be 'cuda' or 'cpu'.")

    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        gen_config = GenerationConfig(
            output_last_hidden_state='generation',
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=512
        )
        responses = self.pipeline(prompts, gen_config=gen_config)
        res = []
        for response in responses:
            res.append({
                'finish_reason': response.finish_reason,
                'hidden_state': response.last_hidden_state
            })
        return res

    def stream_infer(self,
            prompt,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        gen_config = GenerationConfig(
            output_last_hidden_state='generation',
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=512
        )
        responses = self.pipeline.stream_infer([prompt], gen_config=gen_config)
        for response in responses:
            yield {
                'finish_reason': response.finish_reason,
                'hidden_state': response.last_hidden_state
            }
