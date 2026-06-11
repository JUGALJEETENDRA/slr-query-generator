import json
import requests
import sys

# Change: Use command line argument instead of hardcoded path
if len(sys.argv) < 2:
    print("Usage: python generate_benchmark.py <filename>")
    sys.exit(1)

FILE_PATH = f"benchmark/{sys.argv[1]}"
API_URL = "http://localhost:8000/generate"

def run_benchmark():
    # 1. Load the benchmark data
    with open(FILE_PATH, "r") as f:
        data = json.load(f)

    # 2. Iterate and generate
    for item in data:
        print(f"Generating for: {item['id']}...")
        
        payload = {"question": item["question"]}
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 3. Extract the scholar query (adjust the key based on your API response)
            # Assuming your API returns the query string in a field called 'google_scholar'
            # Update this key if your API structure differs!
            item["generated_query"] = result.get("google_scholar", "ERROR_NO_QUERY_FOUND")
            
        except Exception as e:
            print(f"Failed to generate for {item['id']}: {e}")
            item["generated_query"] = "ERROR"

    # 4. Save the updated benchmark file
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=2)
        
    print("Done! benchmark/high_entropy.json has been updated.")

if __name__ == "__main__":
    run_benchmark()