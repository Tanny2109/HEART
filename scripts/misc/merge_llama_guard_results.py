#!/usr/bin/env python3
"""
Merge Llama Guard analysis results from multiple GPU processes into a single CSV file.
"""

import pandas as pd
import glob
import os
import argparse

def merge_results(output_dir, output_filename="llama_guard_analysis_merged.csv"):
    """Merge all rank CSV files into a single file."""
    
    # Find all rank files
    pattern = os.path.join(output_dir, "llama_guard_analysis_rank_*_of_*.csv")
    rank_files = glob.glob(pattern)
    
    if not rank_files:
        print(f"No rank files found in {output_dir} matching pattern: llama_guard_analysis_rank_*_of_*.csv")
        return None
    
    print(f"Found {len(rank_files)} rank files to merge:")
    for f in sorted(rank_files):
        print(f"  - {f}")
    
    # Read and concatenate all files
    dfs = []
    for file in sorted(rank_files):
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged results
    output_path = os.path.join(output_dir, output_filename)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Merged {len(merged_df)} rows into: {output_path}")
    
    # Print summary statistics
    if 'safety_label' in merged_df.columns:
        print("\n📊 Safety Analysis Summary:")
        print(merged_df['safety_label'].value_counts())
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Merge Llama Guard analysis results from multiple processes")
    parser.add_argument("--output_dir", required=True, help="Directory containing the rank CSV files")
    parser.add_argument("--output_filename", default="llama_guard_analysis_merged.csv", 
                       help="Name of the merged output file")
    
    args = parser.parse_args()
    
    merge_results(args.output_dir, args.output_filename)

if __name__ == "__main__":
    main() 