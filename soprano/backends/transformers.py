import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper
from .base import BaseModel


class TransformersModel(BaseModel):
    def __init__(self,
            device='cuda',
            model_path=None,
            **kwargs):
        self.device = device
        
        # Use local model if path provided, otherwise use HuggingFace
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'
        
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
        if temperature <= 0.0:
            temperature = 0.001 # temp must be nonzero
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
        if temperature <= 0.0:
            temperature = 0.001 # temp must be nonzero

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        
        # Prepare Logits Processors for sampling
        logits_processor = LogitsProcessorList()
        if repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        
        logits_warper = LogitsProcessorList()
        if temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature=temperature))
        if top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p=top_p))

        # Helper to sample next token
        def get_next_token(logits, input_seq):
            scores = logits_processor(input_seq, logits)
            scores = logits_warper(input_seq, scores)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            # Sample from the distribution
            return torch.multinomial(probs, num_samples=1)

        with torch.no_grad():
            # Initial forward pass with the prompt
            outputs = self.model(
                input_ids,
                use_cache=True,
                output_hidden_states=True
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # We need to maintain the full sequence for repetition penalty
            generated_ids = input_ids
            
            # Sample the first token
            next_token = get_next_token(next_token_logits, generated_ids)
            
            max_new_tokens = 512
            eos_token_id = self.model.config.eos_token_id
            
            for i in range(max_new_tokens):
                # Append generated token to sequence history
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Run forward pass for the single new token
                outputs = self.model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                # Update cache and get hidden state
                past_key_values = outputs.past_key_values
                current_hidden_state = outputs.hidden_states[-1][:, -1, :] # Last layer, last token
                
                finish_reason = None
                if next_token.item() == eos_token_id:
                    finish_reason = 'stop'
                elif i == max_new_tokens - 1:
                    finish_reason = 'length'

                # Yield result matching lmdeploy format
                yield {
                    'finish_reason': finish_reason,
                    'hidden_state': current_hidden_state
                }
                
                if finish_reason:
                    break
                
                # Prepare for next iteration
                next_token_logits = outputs.logits[:, -1, :]
                next_token = get_next_token(next_token_logits, generated_ids)