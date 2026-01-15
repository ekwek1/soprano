from .vocos.decoder import SopranoDecoder
from .utils.text_normalizer import clean_text
from .utils.text_splitter import split_and_recombine_text
from .utils.auto_select import select_device, select_backend
import torch
import re
from unidecode import unidecode
from scipy.io import wavfile
from huggingface_hub import hf_hub_download
import os
import time


class SopranoTTS:
    """
    Soprano Text-to-Speech model.
    
    Args:
        backend: Backend to use for inference. Options:
            - 'auto' (default): Automatically select best backend. Tries lmdeploy first (fastest),
                               falls back to transformers. CPU always uses transformers.
            - 'lmdeploy': Force use of LMDeploy (fastest, CUDA only)
            - 'transformers': Force use of HuggingFace Transformers (slower, all devices)
        device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps')
        cache_size_mb: Cache size in MB for lmdeploy backend
        decoder_batch_size: Batch size for decoder
    """
    def __init__(self,
            backend='auto',
            device='auto',
            cache_size_mb=100,
            decoder_batch_size=1,
            model_path=None):
        device = select_device(device=device)
        backend = select_backend(backend=backend, device=device)

        if backend == 'lmdeploy':
            from .backends.lmdeploy import LMDeployModel
            self.pipeline = LMDeployModel(device=device, cache_size_mb=cache_size_mb, model_path=model_path)
        elif backend == 'transformers':
            from .backends.transformers import TransformersModel
            self.pipeline = TransformersModel(device=device, model_path=model_path)

        self.device = device
        self.backend = backend
        self.decoder = SopranoDecoder().to(device)
        if model_path:
            decoder_path = os.path.join(model_path, 'decoder.pth')
        else:
            decoder_path = hf_hub_download(repo_id='ekwek/Soprano-1.1-80M', filename='decoder.pth')
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.decoder_batch_size=decoder_batch_size
        self.RECEPTIVE_FIELD = 4 # Decoder receptive field
        self.TOKEN_SIZE = 2048 # Number of samples per audio token

        self.infer("Hello world!") # warmup

    def _preprocess_text(self, texts, min_length=30):
        '''
        adds prompt format and sentence/part index
        Enforces a minimum sentence length by merging short sentences.
        '''
        res = []
        for text_idx, text in enumerate(texts):
            text = text.strip()
            cleaned_text = clean_text(text)
            sentences = split_and_recombine_text(cleaned_text)
            processed = []
            for sentence in sentences:
                processed.append({
                    "text": sentence,
                    "text_idx": text_idx,
                })

            if min_length > 0 and len(processed) > 1:
                merged = []
                i = 0
                while i < len(processed):
                    cur = processed[i]
                    if len(cur["text"]) < min_length:
                        if merged: merged[-1]["text"] = (merged[-1]["text"] + " " + cur["text"]).strip()
                        else:
                            if i + 1 < len(processed): processed[i + 1]["text"] = (cur["text"] + " " + processed[i + 1]["text"]).strip()
                            else: merged.append(cur)
                    else: merged.append(cur)
                    i += 1
                processed = merged
            sentence_idxes = {}
            for item in processed:
                if item['text_idx'] not in sentence_idxes: sentence_idxes[item['text_idx']] = 0
                res.append((f'[STOP][TEXT]{item["text"]}[START]', item["text_idx"], sentence_idxes[item['text_idx']]))
                sentence_idxes[item['text_idx']] += 1
        return res

    def hallucination_detector(self, hidden_state):
        '''
        Analyzes hidden states to find long runs of similar sequences.
        '''
        DIFF_THRESHOLD = 300 # minimal difference between sequences
        MAX_RUNLENGTH = 16 # maximum number of recent similar sequences
        if len(hidden_state) <= MAX_RUNLENGTH: # hidden state not long enough
            return False
        aah_runlength = 0
        for i in range(len(hidden_state) - 1):
            current_sequences = hidden_state[i]
            next_sequences = hidden_state[i + 1]
            diffs = torch.abs(current_sequences - next_sequences)
            total_diff = diffs.sum(dim=0)
            if total_diff < DIFF_THRESHOLD:
                aah_runlength += 1
            elif aah_runlength > 0:
                aah_runlength -= 1
            if aah_runlength > MAX_RUNLENGTH:
                return True
        return False

    def infer(self,
            text,
            out_path=None,
            top_p=0.95,
            temperature=0.0,
            repetition_penalty=1.2,
            retries=0):
        results = self.infer_batch([text],
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            out_dir=None,
            retries=retries)[0]
        if out_path:
            wavfile.write(out_path, 32000, results.cpu().numpy())
        return results

    def infer_batch(self,
            texts,
            out_dir=None,
            top_p=0.95,
            temperature=0.0,
            repetition_penalty=1.2,
            retries=0):
        sentence_data = self._preprocess_text(texts)
        prompts = list(map(lambda x: x[0], sentence_data))
        hidden_states = [None] * len(prompts)
        pending_indices = list(range(0, len(prompts)))
        tries_left = 1 + max(0, retries)
        while tries_left > 0 and pending_indices:
            current_prompts = [prompts[i] for i in pending_indices]
            responses = self.pipeline.infer(current_prompts,
                                            top_p=top_p,
                                            temperature=temperature,
                                            repetition_penalty=repetition_penalty)
            bad_indices = []
            for idx, response in enumerate(responses):
                hidden_state = response['hidden_state']
                hidden_states[pending_indices[idx]] = hidden_state
                if response['finish_reason'] != 'stop':
                    print(f"Warning: A sentence did not complete generation, likely due to hallucination.")
                if retries > 0 and self.hallucination_detector(hidden_state):
                    print(f"Warning: A sentence contained a hallucination.")
                    bad_indices.append(pending_indices[idx])
            if not bad_indices:
                break
            else:
                pending_indices = bad_indices
                tries_left -= 1
                if tries_left > 0:
                    print(f"Warning: {len(pending_indices)} sentence(s) will be regenerated.")
        combined = list(zip(hidden_states, sentence_data))
        combined.sort(key=lambda x: -x[0].size(0))
        hidden_states, sentence_data = zip(*combined)

        num_texts = len(texts)
        audio_concat = [[] for _ in range(num_texts)]
        for sentence in sentence_data:
            audio_concat[sentence[1]].append(None)
        for idx in range(0, len(hidden_states), self.decoder_batch_size):
            batch_hidden_states = []
            lengths = list(map(lambda x: x.size(0), hidden_states[idx:idx+self.decoder_batch_size]))
            N = len(lengths)
            for i in range(N):
                batch_hidden_states.append(torch.cat([
                    torch.zeros((1, 512, lengths[0]-lengths[i]), device=self.device),
                    hidden_states[idx+i].unsqueeze(0).transpose(1,2).to(self.device).to(torch.float32),
                ], dim=2))
            batch_hidden_states = torch.cat(batch_hidden_states)
            with torch.no_grad():
                audio = self.decoder(batch_hidden_states)
            
            for i in range(N):
                text_id = sentence_data[idx+i][1]
                sentence_id = sentence_data[idx+i][2]
                audio_concat[text_id][sentence_id] = audio[i].squeeze()[-(lengths[i]*self.TOKEN_SIZE-self.TOKEN_SIZE):]
        audio_concat = [torch.cat(x).cpu() for x in audio_concat]
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(audio_concat)):
                wavfile.write(f"{out_dir}/{i}.wav", 32000, audio_concat[i].cpu().numpy())
        return audio_concat

    def infer_stream(self,
            text,
            chunk_size=1,
            top_p=0.95,
            temperature=0.0,
            repetition_penalty=1.2):
        start_time = time.time()
        sentence_data = self._preprocess_text([text])

        first_chunk = True
        for sentence, _, _ in sentence_data:
            responses = self.pipeline.stream_infer(sentence,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty)
            hidden_states_buffer = []
            chunk_counter = chunk_size
            for token in responses:
                finished = token['finish_reason'] is not None
                if not finished: hidden_states_buffer.append(token['hidden_state'][-1])
                hidden_states_buffer = hidden_states_buffer[-(2*self.RECEPTIVE_FIELD+chunk_size):]
                if finished or len(hidden_states_buffer) >= self.RECEPTIVE_FIELD + chunk_size:
                    if finished or chunk_counter == chunk_size:
                        batch_hidden_states = torch.stack(hidden_states_buffer)
                        inp = batch_hidden_states.unsqueeze(0).transpose(1, 2).to(self.device).to(torch.float32)
                        with torch.no_grad():
                            audio = self.decoder(inp)[0]
                        if finished:
                            audio_chunk = audio[-((self.RECEPTIVE_FIELD+chunk_counter-1)*self.TOKEN_SIZE-self.TOKEN_SIZE):]
                        else:
                            audio_chunk = audio[-((self.RECEPTIVE_FIELD+chunk_size)*self.TOKEN_SIZE-self.TOKEN_SIZE):-(self.RECEPTIVE_FIELD*self.TOKEN_SIZE-self.TOKEN_SIZE)]
                        chunk_counter = 0
                        if first_chunk:
                            print(f"Streaming latency: {1000*(time.time()-start_time):.2f} ms")
                            first_chunk = False
                        yield audio_chunk.cpu()
                    chunk_counter += 1
