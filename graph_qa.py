import os
import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.llms import Ollama
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
import config

# Set up the LLM model
llm = Ollama(model=config.llm_model_name)

# Get the paths from the config file
output_folder = config.output_folder

# Create a knowledge graph object
graph = NetworkxEntityGraph()

# Load the merged graphml file
combined_graphml_file = config.combined_graphml_file  # e.g., "global_knowledge_graph.graphml"
combined_graphml_path = os.path.join(output_folder, combined_graphml_file)

# Load the .graphml file
loaded_graph = nx.read_graphml(combined_graphml_path)

# Set the loaded graph as the main graph object
graph._graph = loaded_graph

# Create a Q&A chain using GraphQAChain
graph_qa_chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# Question and answering process
def ask_question():
    while True:
        user_input = input("Please enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Answer questions from the knowledge graph
        response = graph_qa_chain.run(user_input)
        print(f"Knowledge graph's response: {response}")
        print(f"Source: {combined_graphml_file}")  # List the loaded graph file as the source

# Start the interaction
ask_question()