import os
import json
import requests
import time
import re

# --- 1. Load Configuration ---
def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"CRITICAL ERROR: config.json not found at: {config_path}")
        return None
        
    with open(config_path, 'r') as f:
        return json.load(f)

# --- 2. Helper: Extract Text from JSON (FIXED FOR NESTED DOCS) ---
def extract_text_from_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text_segments = []
        
        # Case A: Structure {"document": {"pages": [...]}}
        if isinstance(data, dict) and 'document' in data and 'pages' in data['document']:
            print("   [Info] Detected nested 'document' -> 'pages' structure.")
            pages = data['document']['pages']
            for item in pages:
                if 'content' in item:
                    text_segments.append(str(item['content']))
        
        # Case B: Structure {"content": [...]}
        elif isinstance(data, dict) and 'content' in data and isinstance(data['content'], list):
            print("   [Info] Detected direct 'content' list.")
            for item in data['content']:
                if isinstance(item, dict) and 'content' in item:
                    text_segments.append(str(item['content']))
                elif isinstance(item, str):
                    text_segments.append(item)
                    
        # Case C: Simple Dictionary of pages (e.g. "page_1": "...")
        elif isinstance(data, dict):
            for key in sorted(data.keys()):
                if key in ['total_pages', 'content', 'document']: 
                    continue 
                text_segments.append(str(data[key]))
                
        # Case D: List of strings
        elif isinstance(data, list):
            text_segments = [str(item) for item in data]
            
        else:
            text_segments.append(str(data))
            
        return text_segments

    except json.JSONDecodeError:
        print(f"Error: {filepath} is not a valid JSON file.")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

# --- 3. Helper: Clean AI Response ---
def clean_response(text):
    if not text: return ""
    prefixes = [
        r"^Here is the.*?translation.*?:", r"^Sure, here is.*?:", 
        r"^Here is the word-for-word.*?:", r"^Translation:", r"^Here's the translation.*?"
    ]
    suffixes = [
        r"Note: The translation is word-for-word.*", r"Note: I have.*", r"Please let me know.*"
    ]
    for p in prefixes:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    for s in suffixes:
        text = re.sub(s, "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return text

# --- 4. Core: Ollama API Call ---
def translate_segment(text, model_name, url):
    # Prompt designed to prevent summarizing
    if "GTE" in model_name:
        forced_prompt = (
            "Translate the following German text into English. "
            "Provide a complete, accurate translation. "
            "Do NOT summarize. "
            "Do NOT skip any details. "
            "Output ONLY the translated text:\n\n"
            f"{text}"
        )
    elif "GTU" in model_name:
        forced_prompt = (
            "Translate the following German text into Urdu. "
            "Provide a complete, accurate translation. "
            "Do NOT summarize. "
            "Output ONLY the translated text:\n\n"
            f"{text}"
        )
    else:
        forced_prompt = text

    payload = {"model": model_name, "prompt": forced_prompt, "stream": False}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return clean_response(response.json().get('response', ''))
    except Exception as e:
        print(f"\n[ERROR] API Error: {e}")
        return None

# --- 5. Main Execution Flow ---
def main():
    config = load_config()
    if not config: return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, config['global_settings']['data_folder'])
    result_folder = os.path.join(base_dir, config['global_settings']['result_folder'])
    engine_url = config['llm_engine_url']
    
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    print("--- Translation Pipeline Ready ---")

    models = config['models']
    print("\nSelect Translation Mode:")
    for key, model_info in models.items():
        print(f" [{key}] {model_info['name']}")
        
    while True:
        choice = input("\nEnter choice (type 1 or 2): ").strip()
        if choice in models:
            break
        print(f"'{choice}' is not valid. Please type just the number 1 or 2.")

    selected_model = models[choice]
    model_name = selected_model['base_model']
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    if not files:
        print(f"No .json files found in '{data_folder}'.")
        return

    print(f"Found {len(files)} files to process.")

    for filename in files:
        input_path = os.path.join(data_folder, filename)
        output_filename = filename.replace('.json', f"{selected_model['file_suffix']}.txt")
        output_path = os.path.join(result_folder, output_filename)

        if os.path.exists(output_path):
            print(f"[SKIP] {filename} already translated.")
            continue

        print(f"Processing: {filename}...")
        segments = extract_text_from_json(input_path)

        if not segments:
            print("   [WARNING] No text found in file.")
            continue

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("")

        for i, segment in enumerate(segments):
            print(f"   Translating Page {i+1}/{len(segments)}...", end="", flush=True)
            translation = translate_segment(segment, model_name, engine_url)
            
            if translation:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(f"--- Page {i+1} ---\n{translation}\n\n")
                print(" Done.")
            else:
                print(" Failed.")
                if translation is None: break

    print("\nAll jobs completed.")

if __name__ == "__main__":
    main()