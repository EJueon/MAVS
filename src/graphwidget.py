import random
import networkx as nx

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvas

class GraphWidget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)

        plt.style.use('default')
        self.fig = plt.Figure(figsize=(10,7))
            
        self.canvas = FigureCanvas(self.fig)
        self.hLayout=QHBoxLayout()
        self.hLayout.addWidget(self.canvas)

        self.canvas.axes = self.fig.gca(projection='3d')
        self.setLayout(self.hLayout)
        
        self.canvas.draw()
        
    def init_graph(self):
        self.fig.clear()
        self.setLayout(self.hLayout)
        self.canvas.draw()
        
    def init_3dgraph(self):
        self.fig.clear()
        self.canvas.axes = self.fig.gca(projection='3d')

        self.setLayout(self.hLayout)
        self.canvas.draw()
 
    def generate_random_3Dgraph(self, n_nodes: int, radius: int, seed=None):
        if seed is not None:
            random.seed(seed)
        pos={i:(random.uniform(0,1),
        random.uniform(0,1), random.uniform(0,1)) for i in range(n_nodes)}
        G=nx.random_geometric_graph(n_nodes, radius, pos=pos)
        return G

