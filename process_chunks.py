import os
import json
import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.prompts.chat import ChatPromptTemplate
from tqdm import tqdm
import config

# Set the LLM model
llm = Ollama(model=config.llm_model_name)

# Get paths from the config file
json_folder_path = config.json_folder_path
output_folder = config.output_folder

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create ChatPromptTemplate for passing information to the LLM
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Persona:
    You are an expert research assistant specializing in scientific literature analysis for STEM fields, focused on extracting key information from academic papers.

    Goal:
    Given a document, extract all entities and relationships from the following text chunk:

    Steps:
    1. Identify entities with:
       - `entity_name`: Capitalized name of the entity.
       - `entity_type`: Type of entity.
       
    2. Identify related pairs (source_entity, target_entity) with:
       - `source_entity`: Name of the source entity.
       - `target_entity`: Name of the target entity.

    3. Ignore mathematical formulas or symbols.

    4. Return output in clear, structured English in JSON format.

    5. Translate non-English descriptions while keeping everything else intact.
    """),
    ("human", 
     "Extract the following information from the literature chunk titled '{main_title}': \n\ntext: {document_content}")
])

# Use LLMGraphTransformer and pass the prompt
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    # prompt=chat_prompt
)

# Get all files ending with _chunked.json from the folder
json_files = [f for f in os.listdir(json_folder_path) if f.endswith("_chunked.json")]

# Create a global knowledge graph object to store information extracted from all files
global_graph = NetworkxEntityGraph()

# Iterate over each JSON file and process it
for json_file in json_files:
    file_path = os.path.join(json_folder_path, json_file)
    
    # Set output file path
    base_filename = json_file.replace("_chunked", "").replace(".json", "")
    individual_output_file_path = os.path.join(output_folder, f"{base_filename}.graphml")
    
    # Create a local knowledge graph object to store information extracted from this file
    individual_graph = NetworkxEntityGraph()

    # Read the JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading file {json_file}: {e}")
        continue

    # Display progress bar using tqdm
    progress_bar = tqdm(total=len(chunks), desc=f"Processing {json_file}", unit="chunk")

    # Iterate over each chunk and process it
    for chunk_index, chunk in enumerate(chunks):
        try:
            # Create a langchain Document object
            document = Document(page_content=chunk["chunk_content"])
            main_title = chunk["main_title"]  # Use the title string

            # Format the prompt to pass document content to the model
            formatted_prompt = chat_prompt.format(document_content=document.page_content, main_title=main_title)
            # Convert formatted_prompt to a Document object
            prompt_as_document = Document(page_content=str(formatted_prompt))

            # Use LLMGraphTransformer to convert the prompt document to graph documents
            graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents([prompt_as_document])
            
            if not graph_documents_filtered:
                print(f"No graph data extracted from chunk {chunk_index} in {json_file}. Skipping...")
                continue

            # Iterate over the extracted graph documents, updating both local and global graphs
            for graph_document in graph_documents_filtered:
                for node in graph_document.nodes:
                    if node.id not in individual_graph._graph.nodes:
                        individual_graph._graph.add_node(node.id, label=node.id)
                    if node.id not in global_graph._graph.nodes:
                        global_graph._graph.add_node(node.id, label=node.id)

                for edge in graph_document.relationships:
                    if not individual_graph._graph.has_edge(edge.source.id, edge.target.id):
                        individual_graph._graph.add_edge(
                            edge.source.id, 
                            edge.target.id, 
                            relation=edge.type, 
                            weight=edge.weight if hasattr(edge, 'weight') else 1.0
                        )
                    if not global_graph._graph.has_edge(edge.source.id, edge.target.id):
                        global_graph._graph.add_edge(
                            edge.source.id, 
                            edge.target.id, 
                            relation=edge.type, 
                            weight=edge.weight if hasattr(edge, 'weight') else 1.0
                        )
        
        except Exception as e:
            print(f"Error processing chunk {chunk_index} in {json_file}: {e}")
        
        # Update progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Save the local graph to a .graphml file
    try:
        nx.write_graphml(individual_graph._graph, individual_output_file_path)
        print(f"Individual graph saved to {individual_output_file_path}")
    except Exception as e:
        print(f"Error saving individual graph for {json_file}: {e}")

# Save the global graph to a .graphml file
global_output_file_path = os.path.join(output_folder, "global_knowledge_graph.graphml")
try:
    nx.write_graphml(global_graph._graph, global_output_file_path)
    print(f"Global graph saved to {global_output_file_path}")
except Exception as e:
    print(f"Error saving global graph: {e}")