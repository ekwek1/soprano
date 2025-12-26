from .vocos.decoder import SopranoDecoder
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
            Can be overridden with SOPRANO_BACKEND environment variable.
        device: Device to run inference on ('cuda', 'cpu', 'mps')
        cache_size_mb: Cache size in MB for lmdeploy backend
        decoder_batch_size: Batch size for decoder
    
    Environment Variables:
        SOPRANO_BACKEND: Override backend selection ('lmdeploy' or 'transformers')
    """
    def __init__(self,
            backend='auto',
            device='cuda',
            cache_size_mb=10,
            decoder_batch_size=1):
        RECOGNIZED_DEVICES = ['cuda', 'cpu', 'mps']
        RECOGNIZED_BACKENDS = ['auto', 'lmdeploy', 'transformers']
        
        # Check for environment variable override
        env_backend = os.environ.get('SOPRANO_BACKEND')
        if env_backend:
            if env_backend in RECOGNIZED_BACKENDS:
                backend = env_backend
                print(f"Using backend from SOPRANO_BACKEND: {backend}")
            else:
                print(f"Warning: Invalid SOPRANO_BACKEND='{env_backend}'. Must be one of {RECOGNIZED_BACKENDS}. Ignoring.")
        
        assert device in RECOGNIZED_DEVICES, f"unrecognized device {device}, device must be in {RECOGNIZED_DEVICES}"
        
        # Auto-select backend based on device and availability
        if backend == 'auto':
            if device == 'cpu':
                backend = 'transformers'
                print(f"Using backend: {backend} (CPU always uses transformers)")
            else:
                try:
                    import lmdeploy
                    backend = 'lmdeploy'
                    print(f"Using backend: {backend} (lmdeploy available)")
                except ImportError:
                    backend = 'transformers'
                    print(f"Using backend: {backend} (lmdeploy not installed, falling back to transformers)")
        else:
            print(f"Using backend: {backend} (explicitly set)")
        assert backend in RECOGNIZED_BACKENDS, f"unrecognized backend {backend}, backend must be in {RECOGNIZED_BACKENDS}"

        self.device = device
        if backend == 'lmdeploy':
            from .backends.lmdeploy import LMDeployModel
            self.pipeline = LMDeployModel(device=device, cache_size_mb=cache_size_mb)
        elif backend == 'transformers':
            from .backends.transformers import TransformersModel
            self.pipeline = TransformersModel(device=device)

        self.decoder = SopranoDecoder().to(device)
        decoder_path = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
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
            sentences = re.split(r"(?<=[.!?])\s+", text)
            processed = []
            for sentence_idx, sentence in enumerate(sentences):
                old_len = len(sentence)
                new_sentence = re.sub(r"[^A-Za-z !\$%&'*+,-./0123456789<>?_]", "", sentence)
                new_sentence = re.sub(r"[<>/_+]", "", new_sentence)
                new_sentence = re.sub(r"\.\.[^\.]", ".", new_sentence)
                new_sentence = re.sub(r"\s+", " ", new_sentence)
                new_len = len(new_sentence)
                if old_len != new_len:
                    print(f"Warning: unsupported characters found in sentence: {sentence}\n\tThese characters have been removed.")
                new_sentence = unidecode(new_sentence.strip())
                processed.append({
                    "text": new_sentence,
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

    def infer(self,
            text,
            out_path=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        results = self.infer_batch([text],
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            out_dir=None)[0]
        if out_path:
            wavfile.write(out_path, 32000, results.cpu().numpy())
        return results

    def infer_batch(self,
            texts,
            out_dir=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        sentence_data = self._preprocess_text(texts)
        prompts = list(map(lambda x: x[0], sentence_data))
        responses = self.pipeline.infer(prompts,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty)
        hidden_states = []
        for i, response in enumerate(responses):
            if response['finish_reason'] != 'stop':
                print(f"Warning: some sentences did not complete generation, likely due to hallucination.")
            hidden_state = response['hidden_state']
            hidden_states.append(hidden_state)
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
            temperature=0.3,
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
