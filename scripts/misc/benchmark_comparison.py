#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compare performance between HuggingFace and vLLM approaches
"""

import time
import subprocess
import sys
import os
import pandas as pd
import json
from datetime import datetime

def run_benchmark(script_path, args, description):
    """Run a benchmark test and return timing results"""
    print(f"\n🧪 Testing: {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the script
        cmd = [sys.executable, script_path] + args
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! Duration: {duration:.2f}s")
            return {
                "method": description,
                "duration": duration,
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"❌ Failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {
                "method": description,
                "duration": duration,
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
    
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 10 minutes")
        return {
            "method": description,
            "duration": 600,
            "success": False,
            "stdout": "",
            "stderr": "Timeout"
        }
    
    except Exception as e:
        print(f"💥 Exception: {e}")
        return {
            "method": description,
            "duration": 0,
            "success": False,
            "stdout": "",
            "stderr": str(e)
        }

def main():
    """Run comprehensive benchmarks"""
    print("🚀 GPU Inference Performance Comparison")
    print("=" * 60)
    
    # Configuration
    model_path = "/home/tsutar3/HEART/models/SFT/llamaToxic100_hf_v2/"
    csv_path = "/home/tsutar3/HEART/data/insta/type7_version3_output.csv"
    output_dir = "/home/tsutar3/HEART/results/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations
    test_configs = [
        {
            "rows": 10,
            "description": "Small Test (10 rows)"
        },
        {
            "rows": 30,
            "description": "Medium Test (30 rows)"
        }
    ]
    
    results = []
    
    for config in test_configs:
        rows = config["rows"]
        desc = config["description"]
        
        print(f"\n📊 Running {desc}")
        print("-" * 40)
        
        # Test 1: Original HuggingFace approach
        hf_args = [
            "--model_path", model_path,
            "--csv_path", csv_path,
            "--output_path", f"{output_dir}/hf_test_{rows}.csv",
            "--sample_size", str(rows)
        ]
        
        hf_result = run_benchmark(
            "hf_inference.py",
            hf_args,
            f"HuggingFace ({desc})"
        )
        results.append(hf_result)
        
        # Test 2: AutoGen + vLLM + Dynamo-Triton approach (CORRECTED)
        vllm_args = [
            "--model_path", model_path,
            "--csv_path", csv_path,
            "--output_path", f"{output_dir}/autogen_vllm_test_{rows}.csv",
            "--max_rows", str(rows),
            "--tensor_parallel_size", "2",
            "--n_workers", "8",
            "--vllm_port", "8000"
        ]
        
        vllm_result = run_benchmark(
            "vllm_autogen_dynamo.py",
            vllm_args,
            f"AutoGen+vLLM+Dynamo ({desc})"
        )
        results.append(vllm_result)
        
        # Compare results
        if hf_result["success"] and vllm_result["success"]:
            speedup = hf_result["duration"] / vllm_result["duration"]
            print(f"\n📈 Performance Comparison for {desc}:")
            print(f"   HuggingFace: {hf_result['duration']:.2f}s")
            print(f"   vLLM+Dynamo: {vllm_result['duration']:.2f}s")
            print(f"   Speedup: {speedup:.2f}x faster")
            print(f"   Improvement: {(speedup-1)*100:.1f}%")
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        print("✅ Successful Tests:")
        for result in successful_results:
            print(f"   {result['method']}: {result['duration']:.2f}s")
    
    if failed_results:
        print("\n❌ Failed Tests:")
        for result in failed_results:
            print(f"   {result['method']}: {result['stderr']}")
    
    # Calculate averages for comparison
    hf_times = [r["duration"] for r in successful_results if "HuggingFace" in r["method"]]
    vllm_times = [r["duration"] for r in successful_results if "AutoGen+vLLM" in r["method"]]
    
    if hf_times and vllm_times:
        avg_hf = sum(hf_times) / len(hf_times)
        avg_vllm = sum(vllm_times) / len(vllm_times)
        overall_speedup = avg_hf / avg_vllm
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Average HuggingFace: {avg_hf:.2f}s")
        print(f"   Average AutoGen+vLLM+Dynamo: {avg_vllm:.2f}s")
        print(f"   Overall Speedup: {overall_speedup:.2f}x")
        print(f"   Performance Gain: {(overall_speedup-1)*100:.1f}%")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/benchmark_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "results": results,
            "summary": {
                "hf_average": avg_hf if hf_times else 0,
                "vllm_average": avg_vllm if vllm_times else 0,
                "speedup": overall_speedup if hf_times and vllm_times else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Performance projections
    if hf_times and vllm_times:
        print(f"\n📊 Performance Projections:")
        print(f"   1,000 rows:")
        print(f"     HuggingFace: ~{avg_hf * 1000 / 10 / 60:.1f} minutes")
        print(f"     AutoGen+vLLM+Dynamo: ~{avg_vllm * 1000 / 10 / 60:.1f} minutes")
        print(f"   10,000 rows:")
        print(f"     HuggingFace: ~{avg_hf * 10000 / 10 / 60:.1f} minutes")
        print(f"     AutoGen+vLLM+Dynamo: ~{avg_vllm * 10000 / 10 / 60:.1f} minutes")
    
    print("\n🎉 Benchmark completed!")

if __name__ == "__main__":
    main() 