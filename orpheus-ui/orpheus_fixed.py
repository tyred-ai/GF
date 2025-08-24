#!/usr/bin/env python3
"""
Fixed Orpheus Model wrapper that uses correct tokens and disables chunked prefill
"""

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import queue
import threading
import asyncio
from orpheus_tts.decoder import tokens_decoder_sync

class OrpheusModelFixed:
    """Fixed version with proper stop tokens and no chunked prefill"""
    
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = model_name
        self.dtype = dtype
        self.engine = self._setup_engine()
        self.available_voices = ["zoe", "zac", "jess", "leo", "mia", "julia", "leah", "tara"]
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
    
    def _setup_engine(self):
        """Setup engine with conservative memory settings to avoid OOM"""
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            dtype=self.dtype,
            # Disable chunked prefill to avoid OOM
            enable_chunked_prefill=False,
            max_model_len=8192,  # Reduced to save memory
            gpu_memory_utilization=0.75,  # Reduced from 0.85
            max_num_seqs=1,
            enforce_eager=True,  # Disable CUDA graphs to save memory
            enable_prefix_caching=False,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        """Format prompt with correct special tokens"""
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokeniser(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokeniser(prompt, return_tensors="pt")
                start_token = torch.tensor([[128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokeniser.decode(all_input_ids[0])
                return prompt_string
    
    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", 
                            temperature=0.6, top_p=0.8, max_tokens=1200, 
                            stop_token_ids=None, repetition_penalty=1.3):
        """Generate tokens with correct stop tokens"""
        
        # Use correct stop token for Orpheus (not the default 49158)
        if stop_token_ids is None or stop_token_ids == [49158]:
            stop_token_ids = [128258]  # Correct EOS token for Orpheus
        
        # Format the prompt with voice
        prompt_string = self._format_prompt(prompt, voice)
        print(f"Using stop tokens: {stop_token_ids}")
        print(f"Formatted prompt preview: {prompt_string[:100]}...")
        
        # Create sampling params with correct tokens
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )
        
        token_queue = queue.Queue()
        
        async def async_producer():
            async for result in self.engine.generate(
                prompt=prompt_string, 
                sampling_params=sampling_params, 
                request_id=request_id
            ):
                # Place each token text into the queue
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)  # Sentinel to indicate completion
        
        def run_async():
            asyncio.run(async_producer())
        
        thread = threading.Thread(target=run_async)
        thread.start()
        
        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token
        
        thread.join()
    
    def generate_speech(self, **kwargs):
        """Generate speech audio from tokens"""
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))