import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")

def extract_entities_relations(text):
    doc = nlp(text)
    G = nx.DiGraph()
    for ent in doc.ents:
        G.add_node(ent.text, label=ent.label_)
    for token in doc:
        if token.dep_ == "ROOT":
            for child in token.children:
                G.add_edge(token.text, child.text, label=child.dep_)
    return G
