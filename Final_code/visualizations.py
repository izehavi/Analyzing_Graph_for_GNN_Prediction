import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors


class GraphPlotter:
    """
    A class for plotting graphs using an adjacency matrix.
    """

    def __init__(self,W, df_pos=None):
        """
        Initialize the GraphPlotter.

        Args:
            df_pos (pd.DataFrame, optional): DataFrame containing node positions (LATITUDE, LONGITUDE) and others informations about the plot such as node color...
        """
        self.W = W
        self.df_pos = df_pos

    def plot_graph(self, title="Graph", **kwargs):
        """
        Plots a graph from an adjacency matrix using NetworkX.

        Args:
            W (np.ndarray): Adjacency matrix of the graph.
            title (str, optional): Title of the graph plot.
            **kwargs: Additional plotting options.
        """
        G = nx.from_numpy_array(self.W)

        # Use a spring layout if no positions are provided
        positions = nx.spring_layout(G)

        # Plot the graph
        plt.figure(figsize=kwargs.get("figsize", (10, 8)))
        nx.draw(
            G,
            pos=positions,
            with_labels=True,
            node_size=kwargs.get("node_size", 500),
            node_color=kwargs.get("node_color", "blue"),
            edge_color=kwargs.get("edge_color", "gray"),
            font_size=kwargs.get("font_size", 10),
            alpha=kwargs.get("alpha", 0.8),
        )

        # Draw edge weights if W contains weights
        edge_weights = nx.get_edge_attributes(G, "weight")
        if edge_weights:
            nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_weights)

        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_graph_on_map(self, title="Graph on Map"):
        """
        Plots a graph from an adjacency matrix on a geographic map using Basemap.

        Args:
            W (np.ndarray): Adjacency matrix of the graph.
            title (str, optional): Title of the graph plot.
        """
        if self.df_pos is None:
            raise ValueError("df_pos is required to plot a graph on a map.")

        G = nx.from_numpy_array(self.W)

        # Add node positions from the DataFrame
        for idx, (lat, lon) in enumerate(zip(self.df_pos['LATITUDE'], self.df_pos['LONGITUDE'])):
            G.nodes[idx]['pos'] = (lon, lat)

        # Initialize Basemap
        fig, ax = plt.subplots(figsize=(12, 10))
        m = Basemap(projection='merc',
                    llcrnrlat=self.df_pos['LATITUDE'].min() - 2,
                    urcrnrlat=self.df_pos['LATITUDE'].max() + 2,
                    llcrnrlon=self.df_pos['LONGITUDE'].min() - 2,
                    urcrnrlon=self.df_pos['LONGITUDE'].max() + 2,
                    resolution='i', ax=ax)

        m.drawcoastlines()
        m.drawcountries()
        m.fillcontinents(color='gray', alpha=0.3)
        m.drawmapboundary(fill_color='white')

        # Convert positions to map projection
        positions = {node: m(*pos) for node, pos in nx.get_node_attributes(G, 'pos').items()}

        # Plot nodes and edges
        nx.draw_networkx_nodes(G, positions, node_size=500, node_color='blue', alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, positions, font_size=10, font_color='white', ax=ax)

        # Plot edges
        edge_weights = nx.get_edge_attributes(G, "weight")
        if edge_weights:
            edges = G.edges()
            weights = list(edge_weights.values())
            nx.draw_networkx_edges(
                G, positions, edge_color=weights,
                edge_cmap=sns.cubehelix_palette(as_cmap=True),
                edge_vmin=np.min(weights), edge_vmax=np.max(weights),
                width=2, alpha=0.8, ax=ax
            )

        # Add colorbar for edge weights
        norm = mcolors.Normalize(vmin=np.min(weights), vmax=np.max(weights))
        sm = plt.cm.ScalarMappable(cmap=sns.cubehelix_palette(as_cmap=True), norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='horizontal', label='Edge Weight')

        plt.title(title)
        plt.tight_layout()
        plt.show()
