import os
import json
import requests
import re
import sys
import time

# --- 1. Load Configuration ---
def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"CRITICAL ERROR: config.json not found at: {config_path}")
        return None
    with open(config_path, 'r') as f:
        return json.load(f)

# --- 2. Helper: Read Text Files (Prompts/Examples) ---
def read_file_content(filepath):
    """Reads a text file relative to the script directory."""
    if not filepath: return ""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filepath)
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"   [WARNING] File not found: {filepath}")
        return ""

# --- 3. Helper: Extract Text from JSON ---
def extract_text_from_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_segments = []
        
        # Case A: Nested "document" -> "pages"
        if isinstance(data, dict) and 'document' in data and 'pages' in data['document']:
            print("   [Info] Detected structured document (document -> pages).")
            for item in data['document']['pages']:
                if 'content' in item: text_segments.append(str(item['content']))
        
        # Case B: Direct "content" list
        elif isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
            print("   [Info] Detected content list.")
            for item in data['content']:
                if isinstance(item, dict) and 'content' in item: text_segments.append(str(item['content']))
                elif isinstance(item, str): text_segments.append(item)
        
        # Case C: Dictionary of pages
        elif isinstance(data, dict):
            for key in sorted(data.keys()):
                if key not in ['total_pages', 'content', 'document', 'metadata']:
                    text_segments.append(str(data[key]))
        
        # Case D: List
        elif isinstance(data, list):
            text_segments = [str(item) for item in data]
        
        else:
            text_segments.append(str(data))
            
        return text_segments

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

# --- 4. Helper: Clean AI Response ---
def clean_response(text):
    if not text: return ""
    prefixes = [
        r"^Here is the.*?translation.*?:", r"^Sure, here is.*?:", r"^Translation:", 
        r"^Below is the translation.*?:", r"^The translation is as follows.*?:",
        r"^Here is the Urdu translation.*?:", r"^Here is the English translation.*?:"
    ]
    suffixes = [r"Note:.*", r"Please let me know.*", r"I hope this helps.*"]

    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    for s in suffixes:
        text = re.sub(s, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return text

# --- 5. Core: Ollama API Call (Dynamic Prompts) ---
def translate_segment(text, model_config, url):
    model_name = model_config['base_model']
    
    # Load external files
    system_instruction = read_file_content(model_config.get('prompt_file'))
    examples = read_file_content(model_config.get('example_file'))
    
    # Construct the Final Prompt
    # Structure: [System Rules] -> [Examples] -> [Actual Task]
    full_prompt = f"{system_instruction}\n\n"
    
    if examples:
        full_prompt += f"Here are examples of the required style:\n{examples}\n\n"
        
    full_prompt += f"Source Text to Translate:\n{text}"

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_text = response.json().get('response', '')
        return clean_response(raw_text)
    except Exception as e:
        print(f"\n[ERROR] API Error: {e}")
        return None

# --- 6. Main Execution Flow ---
def main():
    config = load_config()
    if not config: return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, config['global_settings']['data_folder'])
    result_folder = os.path.join(base_dir, config['global_settings']['result_folder'])
    engine_url = config['llm_engine_url']
    
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    print("--- Vocational Translation Pipeline (File-Based) ---")

    # User Selection
    models = config['models']
    print("\nSelect Translation Mode:")
    for key, model_info in models.items():
        print(f" [{key}] {model_info['name']}")
        
    while True:
        choice = input("\nEnter choice (type 1 or 2): ").strip()
        if choice in models: break
        print("Invalid choice.")

    selected_model_config = models[choice]
    print(f"\nSelected: {selected_model_config['name']}")
    print(f"Loading Prompt from: {selected_model_config['prompt_file']}")
    print(f"Loading Examples from: {selected_model_config['example_file']}")
    
    # Find Files
    if not os.path.exists(data_folder): return
    files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    if not files: 
        print("No .json files found.")
        return

    print(f"Found {len(files)} files to process.")

    for filename in files:
        input_path = os.path.join(data_folder, filename)
        output_filename = filename.replace('.json', f"{selected_model_config['file_suffix']}.txt")
        output_path = os.path.join(result_folder, output_filename)

        if os.path.exists(output_path):
            print(f"[SKIP] {filename} already translated.")
            continue

        print(f"Processing: {filename}...")
        segments = extract_text_from_json(input_path)
        if not segments: continue

        with open(output_path, 'w', encoding='utf-8') as f: f.write("")

        for i, segment in enumerate(segments):
            print(f"   Translating Page {i+1}/{len(segments)}...", end="", flush=True)
            translation = translate_segment(segment, selected_model_config, engine_url)
            
            if translation:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(f"--- Page {i+1} ---\n{translation}\n\n")
                print(" Saved.")
            else:
                print(" Failed.")
                if translation is None: sys.exit()

    print("\nAll jobs completed.")

if __name__ == "__main__":
    main()