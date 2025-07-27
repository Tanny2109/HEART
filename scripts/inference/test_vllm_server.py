#!/usr/bin/env python3
"""
Simple test script to verify vLLM server can start properly with merged model
"""

import subprocess
import sys
import requests
import time
import os
import signal
import argparse

def test_vllm_server(model_path, port=8000, timeout=60, enable_optimization=True):
    """Test if vLLM server can start and respond"""
    
    print(f"🔍 Testing vLLM server with model: {model_path}")
    print(f"🔧 Port: {port}")
    print(f"⏰ Timeout: {timeout}s")
    print(f"🚀 Optimization: {'Enabled' if enable_optimization else 'Disabled'}")
    
    # Build vLLM server command
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", "1",  # Use single GPU for testing
        "--gpu-memory-utilization", "0.7",
        "--max-model-len", "4096",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--trust-remote-code",
        "--disable-log-stats",
    ]
    
    # Add optimization flags if enabled
    if enable_optimization:
        vllm_cmd.extend([
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "4096",
            "--max-num-seqs", "128",
            "-O", "3",  # Enable torch.compile level 3
        ])
        print("🔥 Using optimized configuration with torch.compile level 3")
    
    print(f"🚀 Starting vLLM server...")
    print(f"Command: {' '.join(vllm_cmd)}")
    
    # Start server
    server_process = subprocess.Popen(
        vllm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                server_ready = True
                break
        except:
            pass
        time.sleep(2)
        print(f"   ... still waiting ({time.time() - start_time:.1f}s)")
    
    if not server_ready:
        print(f"❌ Server failed to start within {timeout}s")
        
        # Get server output
        try:
            stdout, stderr = server_process.communicate(timeout=5)
            print("📄 Server stdout:")
            print(stdout.decode())
            print("📄 Server stderr:")
            print(stderr.decode())
        except:
            print("⚠️ Could not get server output")
        
        # Stop server
        try:
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            else:
                server_process.terminate()
        except:
            pass
        
        return False
    
    print("✅ Server started successfully!")
    
    # Test simple completion
    print("🧪 Testing completion endpoint...")
    try:
        response = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": model_path,
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            completion = result['choices'][0]['text']
            print(f"✅ Completion test successful!")
            print(f"📝 Result: {completion}")
        else:
            print(f"❌ Completion test failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Completion test failed: {e}")
    
    # Test chat endpoint
    print("🧪 Testing chat endpoint...")
    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": model_path,
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            completion = result['choices'][0]['message']['content']
            print(f"✅ Chat test successful!")
            print(f"📝 Result: {completion}")
        else:
            print(f"❌ Chat test failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
    
    # Stop server
    print("🛑 Stopping server...")
    try:
        if hasattr(os, 'killpg'):
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        else:
            server_process.terminate()
        server_process.wait(timeout=5)
    except:
        try:
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            else:
                server_process.kill()
        except:
            pass
    
    print("✅ Test completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test vLLM server with merged model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/tsutar3/HEART/models/SFT/complete_models/llamaToxic100",
        help="Path to the merged model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for server startup"
    )
    parser.add_argument(
        "--no-optimization",
        action="store_true",
        help="Disable optimization flags"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return
    
    # Run the test
    success = test_vllm_server(
        args.model_path, 
        args.port, 
        args.timeout, 
        enable_optimization=not args.no_optimization
    )
    
    if success:
        print("🎉 vLLM server test passed!")
    else:
        print("💥 vLLM server test failed!")
        exit(1)

if __name__ == "__main__":
    main() 