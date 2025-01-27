"""In this document we implement a class object that generalyzing the graph analysis with temporal series"""
import networkx as nx
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from AE_class import Autoencoder
from AE_class_distance import SignalAutoEncoder
#defintion of the class

class GraphSeriesAnalyzer : 
    """__summary__ : This class is used to analyze the graph constructed from the data of the temporal series."""
    def __init__(self, Graphe_builded, nodes_features, name_column: str, Nsize : int, row : int , AE_path ,  cutoff_freq= 2 ,test=False, signals_test = None, plot_result = False, **kwargs):
        
        self.W = Graphe_builded.W
        self.nodes_features = nodes_features
        self.name_column = name_column
        self.Nsize = Nsize
        self.row = row
        self.plot_result = plot_result
        self.cutoff_freq = cutoff_freq
        
        #Compute the graphe
        self.G = Graphe_builded.G
        self.L = Graphe_builded.L.toarray()
        
        # Convert the signals to models
        self.signals = self.get_signals()
        if test :
            signals_array = signals_test[:,self.row: self.row + self.Nsize]
            #Normalization of the signal
            self.signals_mean = np.zeros(signals_array.shape[0])
            self.signals_std = np.zeros(signals_array.shape[0])
            for i in range(signals_array.shape[0]):
                self.signals_mean[i] = np.mean(signals_array[i,:])
                self.signals_std[i] = np.std(signals_array[i,:])
                signals_array[i,:] = (signals_array[i,:] - self.signals_mean[i]) / self.signals_std[i]
            self.signals = signals_array
        else :
            self.signals = self.get_signals()
        self.U, self.lambdas, self.GFT_signal = self.graph_fourier_transform()
        
        # Convert model trend+periodicity
        self.signals_filtered = self.lowpass_filter(cutoff_freq)
        self.t = np.linspace(0, self.Nsize//48, self.Nsize)# abcisse
        self.params, self.fitted_signal = self.fit_signal_with_trend_and_periodic(self.cutoff_freq*1.5)
        
        # Compute the FFT of the signal
        self.signals_fft, self.Lfreqs = self.signal_fft(cutoff_freq)
        self.signals_fft_energyband = self.signal_fft_energyband(15)
        
        # Using AE 
        if name_column == "load" :
            self.encoded_signals = self.Auto_encoder_load_distance()
        else :
            self.encoded_signals = self.Auto_encoder_type(AE_path)
        
        
        #Compute the smoothness generalized
        self.smoothness_model = self.smoothness_calcul_generalized(self.params)
        self.smoothness_fft = self.smoothness_calcul_generalized(self.signals_fft_energyband)
        self.smoothness_AE = self.smoothness_calcul_generalized(self.encoded_signals)
        
        #compute the generalized bandwidth
        self.bandwidth, self.bandwidth_std = self.generalized_bandwidth()
        
        #Compute the local smoothness
        self.Llocal_smoothness_model = self.local_smoothness_generalized(self.params)
        self.Llocal_smoothness_fft = self.local_smoothness_generalized(self.signals_fft_energyband)
        self.Llocal_smoothness_AE = self.local_smoothness_generalized(self.encoded_signals)
        self.smoothness_std_model = np.std(self.Llocal_smoothness_model)
        self.smoothness_std_fft = np.std(self.Llocal_smoothness_fft)
        self.smoothness_std_AE = np.std(self.Llocal_smoothness_AE)
        #print("Done")
        
        

    def get_signals(self):
        """__summary__ : This function is used to extract the signals of the node over the time from the dataframe"""
    
        signals = [] 
        for key in self.nodes_features.keys():
            signals.append(self.nodes_features[key][self.name_column][self.row: self.row + self.Nsize])
        signals_array =np.array(signals, dtype=float)
        
        #Normalization of the signal
        self.signals_mean = np.zeros(signals_array.shape[0])
        self.signals_std = np.zeros(signals_array.shape[0])
        for i in range(signals_array.shape[0]):
            self.signals_mean[i] = np.mean(signals_array[i,:])
            self.signals_std[i] = np.std(signals_array[i,:])
            signals_array[i,:] = (signals_array[i,:] - self.signals_mean[i]) / self.signals_std[i]
        
        return signals_array 
    
    def construct_real_graph(self, **kwargs) :
        """
        Construct a real graph from the weight matrix W.
        """ 
        
        # construction of positions_nodes with the latitude and longitude of the cities
        latitude = kwargs.get("LATITUDE", None)
        Longitude = kwargs.get("LONGITUDE", None)
        positions_nodes = []
        for i in range(len(latitude)):
            positions_nodes.append([latitude[i], Longitude[i]])
            
        G = graphs.Graph(self.W)
        G.set_coordinates(kind=positions_nodes)# TODO mettre les positions des villes
        G.compute_laplacian() 
        
        return G
    
    def graph_fourier_transform(self):
        """_summary_
        Calculate the Fourier Transform of the signal on the graph.
        
        Returns:
            np.ndarray: Fourier Transform of the signal.
        """
        self.G.compute_fourier_basis()
        self.U = self.G.U
        self.lambdas = self.G.e
        self.GFT_signal = self.U.T @ self.signals
        
        return self.U, self.lambdas, self.GFT_signal   
    
    def smoothness_calcul_generalized(self, params) :
        """
        Calculate the smoothness the series over the graph.
        """
        def smoothness_withsum(L_normalized, vec_params_normalized):
            """
            Calcule la régularité d'un signal sur un graphe.
            """
            smoothness = 0  # Initialisation de la régularité
            # Calcul with the sum
            Lsum = [] 
            for i in range (L_normalized.shape[0]):
                smoothness = 0
                for j in range (L_normalized.shape[1]):
                    smoothness -=  L_normalized[i,j] * (vec_params_normalized[i]-vec_params_normalized[j])**2
                Lsum.append(smoothness)
            return smoothness/2
        
        signals_std = self.signals_std[:, np.newaxis]  # Devient (3, 1)
        signals_mean = self.signals_mean[:, np.newaxis]  # Devient (3, 1)

        # Concaténation horizontale
        params = np.concatenate((params, signals_std), axis=1)
        params = np.concatenate((params, signals_mean), axis=1)
        #Normalization of the parameters
        params_normalized = np.zeros_like(params)
        Lsmoothness = []
        for j in range(params.shape[1]):
            params_normalized[:,j] = (params[:,j] - np.mean(params[:,j])) / np.std(params[:,j])
            #print("variance",np.std(self.params[:,j]))
            #print("mean",np.mean(self.params[:,j]))
        
        # Normalisation of the Laplacian
        L = self.G.L

        for j in range (params.shape[1]):
            Lsmoothness.append(params_normalized[:,j].T @ self.L @ params_normalized[:,j])
            
        smoothness = np.sum(np.array(Lsmoothness))/params.shape[1]
        return smoothness

    def local_smoothness_generalized(self, params):
        """ __summary__ : Calculate the local smoothness of temporal series over the graph.
        
        Args:
            params (np.ndarray): Array of parameters of the model that represent the temporal series."""
        L = self.G.L
        signals_std = self.signals_std[:, np.newaxis]  # Devient (3, 1)
        signals_mean = self.signals_mean[:, np.newaxis]  # Devient (3, 1)

        # Concaténation horizontale
        params = np.concatenate((params, signals_std), axis=1)
        params = np.concatenate((params, signals_mean), axis=1)
        #Normalization of the parameters
        params_normalized = np.zeros_like(params)
        for j in range(params.shape[1]):
            params_normalized[:,j] = (params[:,j] - np.mean(params[:,j])) / np.std(params[:,j])
            #print("variance",np.std(self.params[:,j]))
            #print("mean",np.mean(self.params[:,j]))
        

        Llocal_smoothness = np.zeros(L.shape[0])
        for n in range (L.shape[0]):
            local_smoothness = 0
            for i in range (L.shape[0]):
                local_smoothness -= L[n,i] * np.linalg.norm(params_normalized[n,:] -  params_normalized[i,:])**2
            Llocal_smoothness[n] = local_smoothness
        
        
        
        return Llocal_smoothness
             
    def generalized_bandwidth(self):
        """
        Calculate the generalized bandwidth of a time-evolving graph signal.

        Args:
            GFT_signal (np.ndarray): A matrix of size (N, T) where each row represents 
                                    the graph Fourier coefficients over time for a given frequency.
            lambdas (np.ndarray): Array of eigenvalues λ_k of size N, associated with the graph.

        Returns:
            float: Generalized bandwidth of the graph signal.
        """
        GFT_signal = self.GFT_signal
        lambdas = self.lambdas
        U= self.U
        
        N, T = GFT_signal.shape  # N: nombre de fréquences, T: nombre de pas de temps

        # Calcul de ||\hat{f}_k|| pour chaque fréquence k
        norm_f_k = np.sqrt(np.sum(GFT_signal**2, axis=1)) / T

        # Éviter les divisions par zéro
        if np.sum(norm_f_k) == 0:
            return 0
        Lbandwith = []
        for n in range (U.shape[0]):
            # Calcul de la fréquence moyenne lambda_avg(n)
            lambda_avg = np.sum(lambdas *np.abs( norm_f_k * U[n, :])) / np.sum(np.abs(norm_f_k * U[n, :]))

            # Calcul du numérateur et du dénominateur de la largeur de bande
            numerator = np.sum((lambdas - lambda_avg)**2 * norm_f_k**2)
            denominator = np.sum(norm_f_k**2)

            # Calcul final de la largeur de bande généralisée
            bandwidth = np.sqrt(numerator / denominator) if denominator != 0 else 0
            Lbandwith.append(bandwidth)
        Lbandwith = np.array(Lbandwith)
        bandwidth = np.sum(Lbandwith)/U.shape[0]
        bandwith_std = np.std(Lbandwith)
        return bandwidth, bandwith_std
    
    def lowpass_filter(self, cutoff_freq, sample_rate=48 , order=4):
        """
        Applique un filtre passe-bas à des données.

        Paramètres :
        - data : ndarray, le signal à filtrer.
        - cutoff_freq : float, fréquence de coupure (en Hz).
        - sample_rate : float, fréquence d'échantillonnage (en Hz).
        - order : int, ordre du filtre (plus grand = pente plus forte).
        - plot_result : bool, affiche le signal original et filtré si True.

        Retourne :
        - filtered_data : ndarray, le signal filtré.
        """
        plot_result = self.plot_result
        data = self.signals
        # Normalisation de la fréquence de coupure (entre 0 et 1)
        nyquist = 0.5 * sample_rate
        normalized_cutoff = cutoff_freq / nyquist

        # Conception du filtre passe-bas (Butterworth)
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        data_filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            # Application du filtre au signal
            data_filtered[i,:] = signal.filtfilt(b, a, data[i,:])

            # Optionnel : tracer les résultats
            if plot_result and i==0:    
                plt.figure(figsize=(10, 6))
                plt.plot(data[i,:], label="Signal original", alpha=0.7)
                plt.plot(data_filtered[i,:], label=f"Signal filtré (fc={cutoff_freq} Hz)", color='red')
                plt.legend()
                plt.xlabel("Temps (échantillons)")
                plt.ylabel("Amplitude")
                plt.title("Filtrage passe-bas")
                plt.show()

        return data_filtered

    def fit_signal_with_trend_and_periodic(self, cutoff_freq=1.5, poly_degree=2, num_periodic=4):
        """
        Ajuste un modèle en deux étapes : d'abord la tendance polynomiale,
        puis les composantes périodiques après retrait de la tendance.

        Paramètres :
        - cutoff_freq : float, fréquence de coupure pour les bornes des périodiques.
        - poly_degree : int, degré du polynôme pour la tendance.
        - num_periodic : int, nombre de composantes périodiques.

        Retour :
        - params : list, liste des paramètres ajustés pour chaque signal.
        - fitted_signal : ndarray, signal ajusté.
        """
        def trend_model(t, *poly_params):
            """ Modèle pour la tendance polynomiale. """
            return np.array(sum(poly_params[i] * t**i for i in range(len(poly_params))))

        def periodic_model(t, *periodic_params):
            """ Modèle pour les composantes périodiques. """
            periodic = 0
            for i in range(num_periodic):
                amplitude = periodic_params[3 * i]
                freq = periodic_params[3 * i + 1]
                phase = periodic_params[3 * i + 2]
                periodic += amplitude * np.cos(2 * np.pi * freq * t + phase)
            return np.array(periodic)

        plot_result = self.plot_result
        fitted_signal = np.zeros_like(self.signals_filtered)
        t = self.t
        Lparams = []

        for n in range(self.signals_filtered.shape[0]):
            signal = self.signals_filtered[n, :]
            
            # Étape 1 : Ajustement de la tendance
            p0_trend = [np.mean(signal)] + [0] * poly_degree  # Initialisation pour la tendance
            low_bounds_trend = [-np.inf] * len(p0_trend)
            high_bounds_trend = [np.inf] * len(p0_trend)
            p0_trend = np.clip(p0_trend, low_bounds_trend, high_bounds_trend)
            try:
                params_trend, _ = curve_fit(trend_model, t, signal, p0=p0_trend, maxfev=10000, bounds=(low_bounds_trend, high_bounds_trend))
                trend = trend_model(t, *params_trend)  # Tendance ajustée
            except RuntimeError as e:
                print(f"Erreur lors de l'ajustement de la tendance pour le signal {n}: {e}")
                params_trend = p0_trend
                trend = trend_model(t, *params_trend)

            # Retirer la tendance du signal
            detrended_signal = signal - trend

            # Étape 2 : Ajustement des composantes périodiques
            # Estimation initiale des paramètres périodiques
            freqs = np.fft.fftfreq(len(t), d=(t[1] - t[0]))
            fft_magnitudes = np.abs(np.fft.fft(detrended_signal))

            # Filtrer les fréquences positives uniquement
            positive_freqs = freqs[freqs >= 0]
            positive_magnitudes = fft_magnitudes[freqs >= 0]

            # Trier les fréquences positives par ordre décroissant de magnitude
            sorted_indices = np.argsort(positive_magnitudes)[::-1]
            dominant_freqs = positive_freqs[sorted_indices[:num_periodic]]

            p0_periodic = []
            low_bounds_periodic = []
            high_bounds_periodic = []

            for j in range(num_periodic):
                amplitude_est = np.std(detrended_signal)
                freq_est = dominant_freqs[j] if j < len(dominant_freqs) else cutoff_freq / num_periodic
                phase_est = 0

                p0_periodic += [amplitude_est, freq_est, phase_est]
                low_bounds_periodic += [0, 0, -np.pi]
                high_bounds_periodic += [np.inf, cutoff_freq, np.pi]
            
            p0_periodic = np.clip(p0_periodic, low_bounds_periodic, high_bounds_periodic)
            try:
                params_periodic, _ = curve_fit(periodic_model, t, detrended_signal, p0=p0_periodic, maxfev=10000, bounds=(low_bounds_periodic, high_bounds_periodic))
                # Sorted the parameters in order of the frequency
                # Extraire les fréquences et leurs indices associés
                freqs_with_indices = [(params_periodic[3 * i + 1], i) for i in range(num_periodic)]
                sorted_freqs_with_indices = sorted(freqs_with_indices, key=lambda x: x[0])

                # Réorganiser les paramètres périodiques selon l'ordre trié des fréquences
                sorted_params_periodic = []
                for _, original_index in sorted_freqs_with_indices:
                    amplitude = params_periodic[3 * original_index]
                    freq = params_periodic[3 * original_index + 1]
                    phase = params_periodic[3 * original_index + 2]
                    sorted_params_periodic.extend([amplitude, freq, phase])

                params_periodic = np.array(sorted_params_periodic)  # Mise à jour des paramètres triés
                periodic = periodic_model(t, *params_periodic)  # Composantes périodiques ajustées
            except RuntimeError as e:
                print(f"Erreur lors de l'ajustement des composantes périodiques pour le signal {n}: {e}")
                params_periodic = p0_periodic
                periodic = periodic_model(t, *params_periodic)

            # Combiner tendance et périodique pour obtenir le signal ajusté
            fitted_signal[n, :] = trend + periodic
            params = []
            for i in range(len(params_trend)):
                params.append(params_trend[i])
            for j in range(len(params_periodic)):
                params.append(params_periodic[j])
            params = np.array(params)
            Lparams.append(params)

            # Optionnel : afficher les résultats
            if plot_result and n==0:
                plt.figure(figsize=(10, 6))
                plt.plot(t, signal, label="Signal original", alpha=0.7)
                plt.plot(t, fitted_signal[n, :], label="Signal ajusté (modèle complet) ", linestyle="--", color="red")
                plt.legend()
                plt.xlabel("Temps")
                plt.ylabel("Amplitude")
                plt.title(f"Ajustement : Tendance (degré {poly_degree}) + {num_periodic} composantes périodiques")
                plt.show()

        return np.array(Lparams), fitted_signal

    def signal_fft(self, cutoff_freq):
        """
        Compute the FFT of the signal
        """
        N = self.Nsize
        T = 1.0 / 48.0
        yf = np.fft.fft(self.signals_filtered, axis = 1)# compute the fft on the rows
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        
        # Selection of values under the cutoff frequency
        xf_cut = xf[xf <= cutoff_freq]
        yf_cut = yf[:, :xf_cut.shape[0]]/N
        
        if self.plot_result:
            plt.figure(figsize=(10, 6))
            for i in range(yf.shape[0]):
                plt.plot(xf_cut, np.abs(yf_cut[i,:]))
            plt.grid()
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            plt.title("Transformée de Fourier du signal")
            plt.show()
        return np.abs(yf_cut), xf_cut
    
    def signal_fft_energyband(self, nbband):
        """
        This function computes the energy of the signal in the frequency bands.
        It helps to evaluate the smoothness of the signal over the graph and solve frequency inaccuracy issues.

        Args:
            nbband (int): The number of frequency bands to divide the spectrum into.

        Returns:
            np.ndarray: A 2D array where each row corresponds to a node, and each column corresponds to the energy in a specific frequency band.
        """
        
        signals_fft = self.signals_fft  # Spectre du signal (FFT)
        Lfreqs = self.Lfreqs            # Liste des fréquences correspondantes
        energy_band = []                # Liste pour stocker l'énergie par bande

        # Déterminer l'intervalle de fréquence par bande
        max_freq = max(Lfreqs)
        band_width = max_freq / nbband
        Lfmean = []
        # Calculer l'énergie dans chaque bande de fréquence
        for band in range(nbband):
            # Définir la plage de fréquences pour cette bande
            f_start = band * band_width
            f_end = (band + 1) * band_width
            Lfmean.append((f_start + f_end) / 2)
            # Sélectionner les indices des fréquences correspondant à cette bande
            indices = np.where((Lfreqs >= f_start) & (Lfreqs < f_end))[0]
            
            # Calculer l'énergie comme la somme des carrés des amplitudes FFT dans cette bande
            band_energy = np.sum(np.abs(signals_fft[:, indices])**2, axis=1)
            energy_band.append(band_energy)

        # Convertir en array pour retourner les énergies par bandes
        energy_band_array = np.array(energy_band).T

        # Plot les histogrammes pour chaque signal
        if self.plot_result:
            num_signals = energy_band_array.shape[0]
            for i in range(1):
                plt.figure(figsize=(10, 6))
                plt.bar(range(nbband), energy_band_array[i, :], width=0.8, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f"Energy Distribution Across Frequency Bands for Signal {i+1}")
                plt.xlabel("Frequency Band Index")
                plt.ylabel("Energy")
                plt.xticks(range(nbband), [f" {Lfmean[j]:.3f}" for j in range(nbband)])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()

        return energy_band_array
 
    def Auto_encoder_type(self, path_AE) :
        """__summary__ :This function is used to compress the signal with the autoencoder model
        
        Arguments:
        - path_AE : str : the path of the autoencoder model
        - data_type = self.name_column : the type of the data (is used to load the right model)
        
        return : 
        the encoded signal
        """
        
        # Load the autoencoder model
        def load_autoencoder(model_path):
            model = torch.load(model_path, map_location='cpu')
            model.eval()  # Set to evaluation mode
            return model

        # Compress and decompress the signal
        def compress_and_decompress(autoencoder, signal):
            signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0) if signal.ndim == 1 else signal
            with torch.no_grad():
                return autoencoder(signal).squeeze().numpy()

        # Calculate RMSE and MAPE
        def calculate_metrics(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            
            Nbsignals = y_true.shape[0]
            y_true[0] = y_true[0] * self.signals_std[0] + self.signals_mean[0]
            y_pred[0] = y_pred[0] * self.signals_std[0] + self.signals_mean[0]
            for i in range (1):
                plt.clf()
                #plt.plot((y_true[i]-y_pred[i]), label='error')
                plt.plot(y_pred[i], label='output')
                plt.plot(y_true[i], label='input')
                plt.legend()
                plt.show()
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis = 1))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)), axis = 1) * 100
            return rmse, mape

        # Convert the array signal into a torch tensor
        signals = torch.tensor(self.signals_filtered)
        path_AE = path_AE[:-4] +"_"+ self.name_column + ".pth"
        autoencoder = load_autoencoder(path_AE)
        
        if self.plot_result :
            reconstructed_signals = compress_and_decompress(autoencoder, signals)
            rmse, mape = calculate_metrics(signals, reconstructed_signals)
            print(f"RMSE: {np.mean(rmse):.4f} | MAPE: {np.mean(mape):.2f}%")
        
        encoded_signal = autoencoder.encode(torch.tensor(signals)).detach().numpy()
        return encoded_signal
    
    def Auto_encoder_load_distance(self) : 
        """__summary__ : This function is used to compress the signal with the autoencoder model that keeping the distance metric in the embedding space.  """
        
        # I want to load now the model 
        path = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Final_code\AE_model_load_distance.pth"

        modelload = SignalAutoEncoder(self.Nsize, 30, 7)
        modelload.load_state_dict(torch.load(path))
        modelload.eval()
        
        signals_tensor = torch.tensor(self.signals_filtered,dtype=torch.float32 )
        with torch.no_grad():
            signal_output, latent = modelload.forward(signals_tensor)
        if self.plot_result :
            signals_output_array = signal_output.detach().numpy()
            plt.figure(figsize=(10, 6))
            plt.plot(self.t, self.signals_filtered[0, :], label="Signal original", alpha=0.7)
            plt.plot(self.t, signals_output_array[0, :], label="Signal reconstruit", linestyle="--", color="red")
            plt.legend()
            plt.xlabel("Temps")
            plt.ylabel("Amplitude")
            plt.title("Reconstruction du signal avec l'autoencodeur")
            plt.show()
        return latent.detach().numpy()    
    # def Auto_encoder(self, path_AE) : 
    #     # Load the autoencoder model
    #     def load_autoencoder(model_path):
    #         model = torch.load(model_path, map_location='cpu')
    #         model.eval()  # Set to evaluation mode
    #         return model

    #     # Compress and decompress the signal
    #     def compress_and_decompress(autoencoder, signal):
    #         signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0) if signal.ndim == 1 else signal
    #         with torch.no_grad():
    #             return autoencoder(signal).squeeze().numpy()

    #     # Calculate RMSE and MAPE
    #     def calculate_metrics(y_true, y_pred):
    #         y_true, y_pred = np.array(y_true), np.array(y_pred)
            
    #         Nbsignals = y_true.shape[0]
    #         for i in range (Nbsignals):
    #             plt.clf()
    #             plt.plot(np.abs(y_true[i]-y_pred[i]), label='error')
    #             plt.plot(y_true[i], label='input')
    #             plt.legend()
    #             plt.show()
    #         rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis = 1))
    #         mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)), axis = 1) * 100
    #         return rmse, mape

    #     # Convert the array signal into a torch tensor
    #     signals = torch.tensor(self.signals)
    #     autoencoder = load_autoencoder(path_AE)
    #     reconstructed_signals = compress_and_decompress(autoencoder, signals)
    #     rmse, mape = calculate_metrics(signals, reconstructed_signals)
        
    #     encoded_signal = autoencoder.encode(torch.tensor(signals)).detach().numpy()
    #     print(f"RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
    #     return encoded_signal

        
    # def AE (self) : 
    #     """
    #     Load the Autoencoder model and compute the reconstruction error
    #     """
    #     def MAPE(y_true, y_pred):
    #         """
    #         Calcule le MAPE (Mean Absolute Percentage Error) entre les valeurs réelles et prédites.

    #         Arguments :
    #         - y_true (torch.Tensor) : Valeurs réelles (taille N)
    #         - y_pred (torch.Tensor) : Valeurs prédites (taille N)
            
    #         Retour :
    #         - mape (torch.Tensor) : Erreur MAPE (en pourcentage)
    #         """
    #         epsilon = 1e-10  # Pour éviter la division par zéro
    #         y_true = y_true.clamp(min=epsilon)  # Évitez la division par zéro
    #         mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    #         return mape
    #     model = self.model
    #     encoded_signal = model.encode(torch.tensor(self.signals))
        
        
    #     criterion = nn.MSELoss()
    #     signals = torch.tensor(self.signals)
    #     outputs = model.predict(signals)
    #     loss = criterion(outputs, signals)
    #     print("RMSE", loss.item())
        
    #     mape_value  = MAPE(signals, outputs)
    #     print("MAPE", mape_value)
        
    #     return encoded_signal.detach().numpy()