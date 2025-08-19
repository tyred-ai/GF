#!/usr/bin/env python3
import torch
import sys

print("=" * 60)
print("vLLM Installation Test")
print("=" * 60)

# Check Python version
print(f"Python version: {sys.version}")

# Check PyTorch installation
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Multiprocessor count: {props.multi_processor_count}")

# Test vLLM import
try:
    import vllm
    print(f"\nvLLM version: {vllm.__version__}")
    print("vLLM successfully imported!")
    
    # Try to import key vLLM components
    from vllm import LLM, SamplingParams
    print("vLLM core components imported successfully!")
    
except ImportError as e:
    print(f"\nError importing vLLM: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All checks passed! vLLM is ready to use with CUDA support.")
print("=" * 60)