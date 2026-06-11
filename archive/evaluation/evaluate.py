import json
import os

def evaluate_benchmarks():
    # 1. Gather all benchmark files
    benchmark_dir = "benchmark"
    files = [f for f in os.listdir(benchmark_dir) if f.endswith(".json")]
    
    print(f"{'File':<30} | {'ARR':<6} | {'ASS':<6} | {'Failure Mode'}")
    print("-" * 75)

    for file in files:
        path = os.path.join(benchmark_dir, file)
        with open(path, "r") as f:
            data = json.load(f)
            
        total_arr_pass = 0
        total_ass = 0
        loss_counts = {"CONCEPT_ATTRITION": 0, "NARRATIVE_LATCHING": 0, "NONE": 0}

        for item in data:
            query = item.get("generated_query", "").lower()
            concept = item.get("expected_primary", "").lower()
            
            # ARR Logic
            anchor_present = concept in query
            if anchor_present:
                total_arr_pass += 1
            
            # ASS Logic
            anchor_strength = query.count(concept)
            total_ass += anchor_strength
            
            # Loss Type Logic
            if anchor_present:
                if any(word in query for word in ["trend", "challenge", "future", "landscape"]):
                    loss_type = "NARRATIVE_LATCHING"
                else:
                    loss_type = "NONE"
            else:
                loss_type = "CONCEPT_ATTRITION"
            
            loss_counts[loss_type] = loss_counts.get(loss_type, 0) + 1

        # Calculate Averages
        arr_rate = (total_arr_pass / len(data)) * 100
        avg_ass = total_ass / len(data)
        
        print(f"{file:<30} | {arr_rate:>5.1f}% | {avg_ass:>5.2f}  | {loss_counts}")

if __name__ == "__main__":
    evaluate_benchmarks()