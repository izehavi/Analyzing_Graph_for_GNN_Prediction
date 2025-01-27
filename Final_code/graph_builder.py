import numpy as np
import pandas as pd
import random
from pygsp import graphs




class GraphBuilder:
    """
    Class to build a graph from data already calculated by another code.
    """

    def __init__(self,  L_path_to_W, test = False, **kwargs):
        """
        Initializes the graph builder with data and parameters.

        Args:
            L_Ws (list): List of Edges matrices already calculated from another code.
            df_pos (pd.DataFrame): DataFrame containing the positions of the nodes.
            num_nodes (int): Total number of nodes.
            kwargs (dict): Optional parameters.
        """

        self.kwargs = kwargs
        self.W= self.load_Wmatrix(L_path_to_W)
        if test : 
            nbzero = self.count_zeros_in_adjacency_matrix(self.W)
            maxW, minW = np.max(np.abs(self.W)), np.min(np.abs(self.W))
            self.W = self.generate_random_weighted_adjacency_matrix(12, nbzero, minW, maxW)
        self.G = self.construct_real_graph(**kwargs)
        self.L = self.G.L


        
    def load_Wmatrix(self, path_to_W):
        W = np.loadtxt(path_to_W)
        self.W = np.matrix(W)
        np.fill_diagonal(self.W, 0)
        return self.W

    def construct_real_graph(self, **kwargs) :
        """
        Construct a real graph from the weight matrix W.
        """ 
        def normalize_adjacency_matrix(A):
            """
            Normalise la matrice d'adjacence A.
            
            Arguments :
            - A (numpy.ndarray) : Matrice d'adjacence (carrée) de taille (N, N)
            
            Retour :
            - A_norm (numpy.ndarray) : Matrice d'adjacence normalisée de taille (N, N)
            """
            A= np.abs(np.array(A))
            # 1. Calcul du degré de chaque nœud (somme des lignes de la matrice A)
            degrees = np.sum(A, axis=1)  # Degré de chaque nœud (somme des connexions de chaque nœud)
            
            # 2. Évitez la division par zéro (pour les nœuds isolés)
            degrees[degrees == 0] = 1  # Pour éviter la division par zéro
            
            # 3. Calcul de D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            # 4. Normalisation symétrique : A_hat = D^(-1/2) * A * D^(-1/2)
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # Multiplication matricielle
            return A_norm
        
        def real_normalize_adjacency_matrix(A):
            """
            Normalise la matrice d'adjacence A de manière cohérente vis à vis de nos données.
            
            Arguments :
            - A (numpy.ndarray) : Matrice d'adjacence (carrée) de taille (N, N)
            
            Retour :
            - A_norm (numpy.ndarray) : Matrice d'adjacence normalisée de taille (N, N)
            """
            A= np.abs(np.array(A))
            # 1. Calcul du degré de chaque nœud (somme des lignes de la matrice A)
            normA = np.sum(A)
            return A/normA
        
        # construction of positions_nodes with the latitude and longitude of the cities
        latitude = kwargs.get("LATITUDE", None)
        Longitude = kwargs.get("LONGITUDE", None)
        positions_nodes = []
        for i in range(len(latitude)):
            positions_nodes.append([latitude[i], Longitude[i]])
        
        # construction of the graph
        self.W = real_normalize_adjacency_matrix(self.W)
        G = graphs.Graph(self.W)
        G.set_coordinates(kind=positions_nodes)
        G.compute_laplacian() 
        
        return G

    def get_signal(self, name_column : str, row : int): 
        """
        Get the signal of interest to generate the signal graph.
        
        Args:
            self.nodes_features (dict): Dictionary containing the features of the nodes.
            name_column (str): Name of the column containing the signal of interest.
            row (int): Index of the row containing the value of the signal at a given node.
        
        Returns:
            np.ndarray: Vector of the signal.
        """
        
        # calculating the shape of the matrix signals 
        signal = []
        
        for key in self.nodes_features.keys():
            signal.append(self.nodes_features[key][name_column][row])
        signal_array = np.array(signal)
        
        #Normalization of the signal
        signal_array = (signal_array - np.mean(signal_array)) / np.std(signal_array)
        return signal_array

    def graph_fourier_transform(self):
        """_summary_
        Calculate the Fourier Transform of the signal on the graph.
        
        Returns:
            np.ndarray: Fourier Transform of the signal.
        """
        self.G.compute_fourier_basis()
        self.U = self.G.U
        self.lambdas = self.G.e
        self.GFT_signal = self.U.T @ self.signal
        return self.U, self.lambdas, self.GFT_signal

    def keep_top_n(self, N: int) -> np.ndarray:
        """ 
        Keep the N largest values in a matrix and set the others to zero.
        
        Args:
            matrix (np.ndarray): Input matrix.
            N (int): Number of largest values to keep.
        returns:
            np.ndarray: Filtered matrix with only the N largest values.
        """
        matrix = self.W
        # Flatten the matrix to find the N largest values
        flat_vector = matrix.flatten()
        flat_matrix = np.array(flat_vector).flatten()
        
        # If N exceeds the total size of the matrix, keep all elements
        if N >= flat_matrix.shape[0]:
            return matrix
        
        # Find the Nth largest value
        sorted = np.sort(flat_matrix)
        threshold = sorted[-N]
        
        # Create a filtered matrix where only values >= threshold are kept
        filtered_matrix = np.where(matrix >= threshold, matrix, 0)
        
        self.W_filtered = filtered_matrix
        self.W = self.W_filtered
        self.G = self.construct_real_graph(**self.kwargs)
        return filtered_matrix


    def filter_edges_by_energy(self, s):
        """
        Filtre les arêtes d'un graphe pour conserver celles représentant au moins s% de l'énergie totale.

        Args:
            adj_matrix (np.ndarray): Matrice d'adjacence du graphe (symétrique pour un graphe non orienté).
            s (float): Pourcentage d'énergie à conserver (entre 0 et 1).

        Returns:
            filtered_matrix (np.ndarray): Matrice d'adjacence filtrée avec uniquement les arêtes sélectionnées.
            total_energy (float): Énergie totale calculée avant filtrage.
            selected_energy (float): Énergie cumulée des arêtes retenues.
        """
        
        # adj_matrix
        adj_matrix = self.W
        # Calculer l'énergie totale (racine carrée de la somme des carrés des poids)
        total_energy = np.sum(adj_matrix ** 2)
        #total_energy = np.sum(adj_matrix)
        # Calculer l'énergie relative pour chaque lien (poids / énergie totale)
        matrix = adj_matrix**2 / total_energy
        
        flat_vector = matrix.flatten()
        flat_matrix = np.array(flat_vector).flatten()
        
        
        # Find the Nth largest value
        sorted = np.sort(flat_matrix)

        # on cherche le seuil correspondant à s% d'énergie
        i=0
        if s<1 :
            cumulative_energy = 0
            while cumulative_energy <= s :
                cumulative_energy += sorted[-i]
                i+=1
            threshold = sorted[-i]
        print("cumulative_energy", cumulative_energy)
        # Create a filtered matrix where only values >= threshold are kept
        filtered_matrix = np.where(matrix >= threshold, self.W, 0)
        
        self.W_filtered = filtered_matrix
        self.W = self.W_filtered
        self.G = self.construct_real_graph(**self.kwargs)
        
        return self.W_filtered
    
    def kruskal_maximal_spanning_tree(self):
        """
        Implements Kruskal's algorithm for a maximal spanning tree (strongest edges).

        Returns:
            np.ndarray: Adjacency matrix representing the maximal spanning tree.
        """
        n = self.W.shape[0]  # Number of nodes
        edges = []

        # Build the list of edges with their weights
        for i in range(n):
            for j in range(i + 1, n):  # Undirected graph (avoid double counting)
                if self.W[i, j] != 0:
                    edges.append((self.W[i, j], i, j))

        # Sort edges by weight in descending order (for maximal spanning tree)
        edges.sort(reverse=True, key=lambda x: x[0])

        # Union-Find data structure for cycle detection
        parent = list(range(n))
        rank = [0] * n

        def find(node):
            """ Find the representative of the set (with path compression). """
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]

        def union(node1, node2):
            """ Merge two sets (with rank optimization). """
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                # Attach smaller tree under the larger one
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    rank[root1] += 1
                return True  # The merge was successful (no cycle)
            return False  # Nodes already connected (cycle detected)

        # Initialize the maximal spanning tree adjacency matrix
        mst_matrix = np.zeros((n, n))
        edges_used = 0

        # Iterate through edges and build the maximal spanning tree
        for weight, u, v in edges:
            if edges_used == n - 1:  # Stop when the tree has n-1 edges
                break
            if union(u, v):
                mst_matrix[u, v] = weight
                mst_matrix[v, u] = weight  # Undirected graph
                edges_used += 1

        # Store the result in the filtered matrix attribute
        self.W_filtered = mst_matrix
        self.W= self.W_filtered
        self.G = self.construct_real_graph(**self.kwargs)
        return mst_matrix

    def count_zeros_in_adjacency_matrix(self, adj_matrix):
        """
        Compte le nombre de zéros dans une matrice d'adjacence.

        Args:
            adj_matrix (np.ndarray): Matrice d'adjacence (carrée).

        Returns:
            int: Nombre de zéros dans la matrice (hors diagonale).
        """
        if not isinstance(adj_matrix, np.ndarray):
            raise ValueError("La matrice d'adjacence doit être un tableau NumPy.")
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("La matrice d'adjacence doit être carrée.")

        num_nodes = adj_matrix.shape[0]
        # Compter les zéros hors de la diagonale
        zero_count = np.sum(adj_matrix == 0) - num_nodes

        return zero_count

    def generate_random_weighted_adjacency_matrix(self, num_nodes, num_zeros, min_weight=0.01, max_weight=2):
        """
        Génère une matrice d'adjacence pondérée aléatoire avec un nombre spécifique de zéros.

        Args:
            num_nodes (int): Nombre de nœuds dans le graphe (dimension de la matrice).
            num_zeros (int): Nombre de zéros dans la matrice d'adjacence (hors diagonale).
            min_weight (int): Poids minimum des arêtes.
            max_weight (int): Poids maximum des arêtes.

        Returns:
            np.ndarray: Matrice d'adjacence pondérée aléatoire.
        """
        if num_zeros > num_nodes * (num_nodes - 1):
            raise ValueError("Le nombre de zéros est trop grand pour la matrice donnée.")

        # Créer une matrice pleine de poids aléatoires
        adj_matrix = np.random.uniform(min_weight, max_weight, size=(num_nodes, num_nodes))

        # Rendre la matrice symétrique
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        # Placer des zéros aléatoirement hors de la diagonale
        indices = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
        random.shuffle(indices)

        zero_indices = indices[:num_zeros // 2]
        for i, j in zero_indices:
            adj_matrix[i, j] = 0
            adj_matrix[j, i] = 0

        # Mettre les zéros sur la diagonale
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix

    
    def generate_line_graph(self, num_nodes=12):
        """
        Generate a line graph with a specified number of nodes.

        Args:
            num_nodes (int): Number of nodes in the graph.

        Returns:
            np.ndarray: Adjacency matrix of size (num_nodes, num_nodes).
        """
        # Initialize an empty adjacency matrix
        A = np.zeros((num_nodes, num_nodes), dtype=int)

        # Assign weights or binary edges for a line graph
        A[0, num_nodes - 1] = 1
        A[num_nodes - 1, 0] = 1
        for i in range(num_nodes - 1):
            A[i, i + 1] = 1
            A[i + 1, i] = 1
        
        self.W = np.matrix(A)
        self.G = self.construct_real_graph(**self.kwargs)
        self.L = self.G.L






