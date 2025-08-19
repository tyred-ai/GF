#!/usr/bin/env python3
"""
Simple vLLM inference test to verify the installation works correctly.
Uses a small model to test basic functionality.
"""

from vllm import LLM, SamplingParams

def test_vllm_inference():
    print("Testing vLLM inference with a small model...")
    print("-" * 60)
    
    # Use a small model for testing (Qwen 0.5B is tiny and fast)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading model: {model_name}")
    print("This may take a moment on first run to download the model...")
    
    try:
        # Initialize the LLM
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=2048,  # Limit context for faster testing
            gpu_memory_utilization=0.5,  # Use only half GPU memory
        )
        
        # Prepare a simple prompt
        prompts = [
            "What is the capital of France?",
            "Complete this sentence: The quick brown fox"
        ]
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=50,
        )
        
        print("\nRunning inference...")
        print("-" * 60)
        
        # Generate responses
        outputs = llm.generate(prompts, sampling_params)
        
        # Print the outputs
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {generated_text.strip()}")
            print("-" * 40)
        
        print("\n✅ vLLM inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    success = test_vllm_inference()
    exit(0 if success else 1)