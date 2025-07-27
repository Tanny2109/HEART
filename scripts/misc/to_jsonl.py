import pandas as pd
import json
from pathlib import Path

# File paths
csv_files = [
    "sft1.csv", "sft2.csv", "sft3.csv"
]

output_paths = []

# Convert each CSV file to JSONL format with 'prompt' and 'response' keys
for file_name in csv_files:
    csv_path = f"/home/tsutar3/HEART/data/{file_name}"
    jsonl_path = f"/home/tsutar3/HEART/data/{file_name.replace('.csv', '.jsonl')}"
    output_paths.append(jsonl_path)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure prompt/response columns exist
    if 'prompt' in df.columns and 'completion' in df.columns:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json.dump({"prompt": str(row["prompt"]), "completion": str(row["completion"])}, f)
                f.write("\n")

output_paths
