#!/usr/bin/env python3
"""
Quick test script for HF inference
"""

import os
import sys
import subprocess

def main():
    # Test with a small sample
    print("Testing HF inference script with sample data...")
    
    # Change to HEART directory
    os.chdir("/home/tsutar3/HEART")
    
    # Run the inference script with a small sample
    cmd = [
        "python", "scripts/hf_inference.py",
        "--model_path", "~/HEART/models/SFT/complete_models/empathy_15",
        "--csv_path", "data/insta/type7_version3_output.csv",
        "--sample_size", "2",
        "--output_path", "test_results.csv"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("Return code:", result.returncode)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            if os.path.exists("test_results.csv"):
                print("✅ Output file created successfully!")
            else:
                print("❌ Output file not found")
        else:
            print("❌ Test failed with return code:", result.returncode)
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running test: {e}")

if __name__ == "__main__":
    main() 