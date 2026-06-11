# extract_audit_sample.py
import json
import random

INPUT_TELEMETRY = "term_telemetry.json"
OUTPUT_AUDIT_SHEET = "manual_audit_workspace.json"

def generate_audit_cohort():
    try:
        with open(INPUT_TELEMETRY, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {INPUT_TELEMETRY}. Run the benchmark script first.")
        return

    pending_pool = [entry for entry in data if entry.get("classification") == "pending_manual_audit"]
    print(f"📋 Total terms inside pending_manual_audit pool: {len(pending_pool)}")
    
    if len(pending_pool) < 100:
        print("⚠️ Warning: Pending pool has fewer than 100 terms. Sampling the entire pool instead.")
        sampled_cohort = pending_pool
    else:
        sampled_cohort = random.sample(pending_pool, 100)

    workspace_data = []
    for item in sampled_cohort:
        workspace_data.append({
            "question_id": item["question_id"],
            "generated_term": item["term"],
            "my_manual_classification": ""
        })

    with open(OUTPUT_AUDIT_SHEET, "w") as f:
        json.dump(workspace_data, f, indent=2) # 💡 FIX: Removed the broken assignment
        
    print(f"✅ Success! Created {OUTPUT_AUDIT_SHEET} with {len(workspace_data)} random terms.")

if __name__ == "__main__":
    generate_audit_cohort()