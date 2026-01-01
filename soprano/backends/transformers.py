import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self,
            device='cuda',
            model_path=None,
            **kwargs):
        self.device = device
        
        # Use local model if path provided, otherwise use HuggingFace
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-80M'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()

    def infer(self,
            prompts,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=512,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        res = []
        eos_token_id = self.model.config.eos_token_id
        for i in range(len(prompts)):
            seq = outputs.sequences[i]
            hidden_states = []
            num_output_tokens = len(outputs.hidden_states)
            for j in range(num_output_tokens):
                token = seq[j + seq.size(0) - num_output_tokens]
                if token != eos_token_id: hidden_states.append(outputs.hidden_states[j][-1][i, -1, :])
            last_hidden_state = torch.stack(hidden_states).squeeze()
            finish_reason = 'stop' if seq[-1].item() == eos_token_id else 'length'
            res.append({
                'finish_reason': finish_reason,
                'hidden_state': last_hidden_state
            })
        return res

    def stream_infer(self,
            prompt,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        raise NotImplementedError("transformers backend does not currently support streaming, please consider using lmdeploy backend instead.")
