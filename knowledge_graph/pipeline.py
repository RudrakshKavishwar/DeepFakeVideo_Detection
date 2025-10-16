from .extractor import extract_entities_relations
from .visualization import visualize_graph

def build_and_visualize_kg(segments, output_path="data/graph.html"):
    import networkx as nx
    G = nx.DiGraph()

    for seg in segments:
        caption = seg["caption"]
        subgraph = extract_entities_relations(caption)
        G = nx.compose(G, subgraph)

    path = visualize_graph(G, output_path)
    return {"nodes": len(G.nodes()), "edges": len(G.edges())}, path
