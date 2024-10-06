import networkx as nx
import matplotlib.pyplot as plt

graph = nx.read_graphml("global_knowledge_graph.graphml")

target_node = "PaLM"
if target_node in graph:
    neighbors = list(graph.neighbors(target_node))
    max_neighbors = 10
    neighbors = neighbors[:max_neighbors] if len(neighbors) > max_neighbors else neighbors
    subgraph_nodes = [target_node] + neighbors
    subgraph = graph.subgraph(subgraph_nodes)
else:
    print(f"Node '{target_node}' not found in the graph.")
    subgraph = None

if subgraph:
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph, seed=42)

    nx.draw_networkx_nodes(subgraph, pos, node_size=[1000 if n == target_node else 500 for n in subgraph.nodes()],
                           node_color=['red' if n == target_node else 'skyblue' for n in subgraph.nodes()],
                           alpha=0.8)

    nx.draw_networkx_edges(subgraph, pos, width=2, alpha=0.5, edge_color='gray')

    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_color='black')

    plt.title(f"Top {max_neighbors} Connections of '{target_node}'")
    plt.axis("off")
    plt.show()
else:
    print(f"Unable to create subgraph for '{target_node}'.")