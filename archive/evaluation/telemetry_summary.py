import json
import os
import glob

def summarize():
    # Looks for all .json files in the 'benchmark' folder
    benchmark_dir = "benchmark"
    files = glob.glob(os.path.join(benchmark_dir, "*.json"))
    
    total_samples = 0
    total_arr_pass = 0
    
    print("--- AUTOMATED TELEMETRY SUMMARY ---")
    
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            for item in data:
                total_samples += 1
                query = item.get("generated_query", "").lower()
                concept = item.get("expected_primary", "").lower()
                
                # ARR Logic
                if concept in query:
                    total_arr_pass += 1
    
    if total_samples > 0:
        arr_pct = (total_arr_pass / total_samples) * 100
        print(f"Total Samples Processed: {total_samples}")
        print(f"Aggregate ARR: {arr_pct:.1f}%")
    else:
        print("No samples found.")
    print("-----------------------------------")

if __name__ == "__main__":
    summarize()