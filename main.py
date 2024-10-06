# main.py

import subprocess
import sys

def run_script(script_name):
    """
    Runs a Python script using subprocess.
    """
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the workflow.
    """
    print("Step 1: Preprocessing PDFs...")
    run_script("preprocess_pdfs.py")
    
    print("Step 2: Building Knowledge Graphs...")
    run_script("process_chunks.py")
    
    print("Step 3: Starting Q&A on Knowledge Graph...")
    run_script("graph_qa.py")

if __name__ == "__main__":
    main()