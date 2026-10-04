import json
import time
from datasets import load_dataset
from datasets import DownloadConfig
import re

print("Downloading 'emilgoh/verilog-dataset-v2' from Hugging Face...")

# Add network resilience for unstable connections
max_retries = 5
data = None

for attempt in range(max_retries):
    try:
        print(f"Download attempt {attempt + 1} of {max_retries}...")
        
        # Increase the timeout and force retries on the HTTP level
        dl_config = DownloadConfig(max_retries=10)
        data = load_dataset("emilgoh/verilog-dataset-v2", split="train", download_config=dl_config)
        
        print("Download successful!")
        break # Exit the retry loop if successful
        
    except Exception as e:
        print(f"Attempt {attempt + 1} failed. Error: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...\n")
            time.sleep(5)
        else:
            print("All download attempts failed. Check your network stability.")
            exit()

# ... [Keep the rest of your bug injection logic below here exactly the same] ...