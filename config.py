# config.py

# Path Configuration
input_folder = "/Users/tanqiwen/Documents/GraphRAG-Local"  # Folder containing input PDF files
output_folder = "/Users/tanqiwen/Documents/GraphRAG-Local"  # Output folder
json_folder_path = output_folder  # Folder to store chunked JSON files
combined_graphml_file = "global_knowledge_graph.graphml"  # Name of the combined graph file

# Text Splitting Parameters
chunk_size = 1200  # Size of text chunks
chunk_overlap = 400  # Overlap size between text chunks

# LLM Model Parameters
llm_model_name = "llama3.2:latest"  # Name of the LLM model to use, can be modified as needed