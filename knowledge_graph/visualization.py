import networkx as nx
import plotly.graph_objects as go

def visualize_graph(G, output_path="graph.html"):
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"),
                            hoverinfo="none", mode="lines")

    node_x, node_y, text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                            text=text, textposition="top center",
                            marker=dict(size=10, color="skyblue"))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False))
    fig.write_html(output_path)
    return output_path
