#%% Importation of the libraries
import torch
import torch.nn as nn  # Import nn to create the network layers
import torch.optim as optim  # Import optim to define the optimizer
from torch.utils.data import DataLoader, TensorDataset  # DataLoader to load and batch the data
from dataloader import DataLoader as DL
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from torchsummary import summary

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Autoencoder(nn.Module):
    def __init__(self, nsize=1000, latent_size=32, deepness=3):
        super(Autoencoder, self).__init__()
        
        # Calcul du facteur pour réduire la taille
        factor = (nsize / latent_size) ** (1 / deepness)  # Réduction progressive
        
        #  Encoder
        self.encoder_layers = []
        for i in range(deepness - 1):
            in_size = int(round(nsize / factor**(i)))
            out_size = int(round(nsize / factor**(i + 1)))
            self.encoder_layers.append(nn.Linear(in_size, out_size))
            self.encoder_layers.append(nn.ReLU())
        self.encoder_layers.append(nn.Linear(int(round(nsize / factor**(deepness - 1))), latent_size))
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        #  Decoder
        self.decoder_layers = []
        for i in range(deepness - 1):
            in_size = int(round(latent_size * factor**(i)))
            out_size = int(round(latent_size * factor**(i + 1)))
            self.decoder_layers.append(nn.Linear(in_size, out_size))
            self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Linear(int(round(latent_size * factor**(deepness - 1))), nsize))
        self.decoder = nn.Sequential(*self.decoder_layers)
        
    def forward(self, x):
        x= x.float()
        x = self.encoder(x)  # Compress input
        #x = (x - torch.mean(x)) / torch.std(x)
        x = self.decoder(x)  # Reconstruct input
        
        x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def encode(self, x):
        x= x.float()
        x = self.encoder(x)
        x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def decode(self, x):
        x= x.float()
        x = self.decoder(x)
        #x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def predict(self, x):
        x= x.float()
        x = self.encoder(x)
        meanx = torch.mean(x)
        stdx = torch.std(x)
        x = (x - meanx) / stdx
        x = self.decoder(x)
        x = x * stdx + meanx
        return x



#%% Variables global
nsize = 1000
deepness = 5
latent_size = 16


batch_size = 256 # Number of samples in each batch
num_epochs = 100  # Number of epochs to train
path_train = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\train.csv"
path_test = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\test.csv"

#%%2 Importation of the dataset
data = DL(path_train, path_test, kwargs={"start_date": "2018-01-01", "end_date": None})
nodes_dataframe = data.nodes_dataframe

def get_signals(nodes_features,Lname_column, Nsize):
    """__summary__ : This function is used to extract the signals of the node over the time from the dataframe"""
    def lowpass_filter(data, cutoff_freq, plot_result = False,  sample_rate=48 , order=4):
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
            if plot_result:
                plt.figure(figsize=(10, 6))
                plt.plot(data[i,:], label="Signal original", alpha=0.7)
                plt.plot(data_filtered[i,:], label=f"Signal filtré (fc={cutoff_freq} Hz)", color='red')
                plt.legend()
                plt.xlabel("Temps (échantillons)")
                plt.ylabel("Amplitude")
                plt.title("Filtrage passe-bas")
                plt.show()

        return data_filtered
    signals = [] 
    Ntotal = len(nodes_features[list(nodes_features.keys())[0]][Lname_column[0]])
    for name_column in Lname_column:
        for i in range(Ntotal // Nsize):
            for key in nodes_features.keys():
                signals.append(np.array(nodes_features[key][name_column][Nsize*i :Nsize * (i+1)]))
    signals_array = np.array(signals)
    
    signals_array = lowpass_filter(signals_array, 3, plot_result=False)
    
    signal_array_normalized = np.zeros(signals_array.shape)
    #Normalization of the signal
    for i in range(signals_array.shape[0]):
        signal_array_normalized[i,:] = (signals_array[i,:] - np.mean(signals_array[i,:])) / np.std(signals_array[i,:])
    
    return signal_array_normalized

#signals = get_signals(data.nodes_dataframe, ["load","temp","nebu","wind", "tempMax","tempMin"], nsize)
signals = get_signals(data.nodes_dataframe, ["tempMax"], nsize)
print(signals.shape)
dataset_tensor = torch.tensor(signals)
dataloader = DataLoader(TensorDataset(dataset_tensor), batch_size=batch_size, shuffle=True)



#%% 3️ Create the model, define loss and optimizer
model = Autoencoder(nsize,latent_size, deepness )  # Create an instance of the Autoencoder
criterion = nn.MSELoss()  # Mean Squared Error is used for reconstruction loss
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 3e-4,
                             weight_decay = 1e-8)  # Use Adam optimizer

summary(model,input_size=(nsize,)) 

#%% 4️ Train the model
Lloss = []
for epoch in range(num_epochs):
    i=0
    for batch in dataloader:
        signals = batch[0].float()  # Les signaux sont dans la première position
        optimizer.zero_grad()
        outputs = model.forward(signals)
        loss = criterion(outputs, signals)
        Lloss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if i==0 :
            signal_input = batch[0][0]
            signal_output = model.forward(signal_input)
            plt.clf()
            plt.plot(signal_input, label='input')
            plt.plot(signal_output.detach().numpy(), label='output')
            plt.legend()
            plt.show()
            i+=1

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

plt.clf()
plt.plot(Lloss)
plt.show()
print("Training complete!")

# %%
torch.save(model, "AE_model.pth")
# %%
model2 = torch.load("AE_model.pth")
# %%
size = 48
signal_input = dataset_tensor[70]
signal_output = model2.forward(signal_input)
plt.plot(signal_input[:size], label='input')
plt.plot(signal_output.detach().numpy()[:size], label='output')
plt.legend()
plt.show()
# %%
print(model2.encode(signal_input))
# %%
