import json
import os
import time
import google.generativeai as genai

# --- Configuration ---
# This script automatically finds files in its own directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_CORPUS_INPUT = os.path.join(SCRIPT_DIR, "application_corpus.json")
# CORRECTED: Pointing to the original knowledge corpus file as you specified
KNOWLEDGE_CORPUS_INPUT = os.path.join(SCRIPT_DIR, "knowledge_corpus.json")
ALIGNED_CORPUS_OUTPUT = os.path.join(SCRIPT_DIR, "application_corpus_v2_aligned.json")

# SECURE: Load API key from environment variables
# Before running, set this in your terminal: setx GOOGLE_API_KEY "YOUR_NEW_API_KEY"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_json(file_path: str):
    """Loads a JSON file from the given path."""
    print(f"Loading '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'. Exiting.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_links_from_llm(problem: str, solution: str, knowledge_context: str) -> list:
    """
    Uses a Large Language Model to determine which knowledge points are used.
    """
    prompt = f"""
    You are an expert in mathematical problem solving and data analysis. Your task is to analyze a math problem and its solution and identify ALL the specific mathematical concepts and formulas used from a predefined list.

    Here is the list of all available knowledge points with their IDs:
    ---
    {knowledge_context}
    ---

    Here is the problem and its solution:
    Problem: "{problem}"
    Solution: "{solution}"

    Based on the problem and solution, return a JSON list containing the `knowledge_id`s of ONLY the concepts that were explicitly or implicitly used to arrive at the answer. Do not guess or include concepts that are not used. The output must be ONLY the JSON list and nothing else.
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
        
    except Exception as e:
        print(f"    [ERROR] An error occurred during the LLM call: {e}")
        return []

def main():
    """Main script to realign the application corpus."""
    print("--- Starting Corpus Re-alignment Process ---")

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set. Please set it and restart your terminal.")
        return
        
    genai.configure(api_key=GOOGLE_API_KEY)
    
    app_corpus = load_json(APP_CORPUS_INPUT)
    knowledge_corpus = load_json(KNOWLEDGE_CORPUS_INPUT)
    
    if not app_corpus or not knowledge_corpus:
        return
        
    knowledge_context = "\n".join([f"- {k['knowledge_id']}: {k['title']}" for k in knowledge_corpus])
    
    aligned_app_corpus = []
    total_apps = len(app_corpus)

    print(f"\nFound {total_apps} application points to process.")
    
    for i, app in enumerate(app_corpus):
        print(f"Processing application {i+1}/{total_apps} (ID: {app['app_id']})...")
        
        new_links = get_links_from_llm(app['problem'], app['solution'], knowledge_context)
        
        app['knowledge_links'] = new_links
        aligned_app_corpus.append(app)
        
        print(f"    -> Linked to: {new_links}")
        
        time.sleep(0.5)

    with open(ALIGNED_CORPUS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(aligned_app_corpus, f, indent=4)
        
    print(f"\n--- Re-alignment Complete ---")
    print(f"Successfully saved the new aligned corpus to '{ALIGNED_CORPUS_OUTPUT}'.")

if __name__ == "__main__":
    main()