import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class graphVisualization:

    def simple_graph():

        G = nx.Graph()
        G.add_edge('a','b')
        G.add_edge('a','c')
        G.add_edge('a','d')
        G.add_edge('a','e')
        G.add_edge('b','c')
        G.add_edge('b','d')
        G.add_edge('b','e')
        G.add_edge('e','c')

        return nx.info(G)
