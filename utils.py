import numpy as np
import torch 
import h5py # pour gérer les formats de données utilisés ici 
import torch
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Définition d'un Dataset personnalisé
class MyDataset(torch.utils.data.Dataset):
    """
    Classe personnalisée pour gérer un ensemble de données à partir d'un fichier HDF5.
    """

    def __init__(self, chemin_vers_donnees):
        """
        Initialise le jeu de données en chargeant les signaux, les SNR et les étiquettes.

        Paramètres :
        - chemin_vers_donnees : chemin vers le fichier HDF5 contenant les données.
        """
        # Chargement des données à partir d'un fichier HDF5
        donnees = h5py.File(chemin_vers_donnees, 'r')
        self.signaux = np.array(donnees['signaux']).transpose(0, 2, 1)  # Transpose pour le bon format
        self.snr = np.array(donnees['snr'])  # SNR (Rapport Signal-Bruit)
        self.etiquettes_id = np.array(donnees['labels'])  # Identifiants des étiquettes
        self.noms_etiquettes = get_labels(donnees)  # Récupération des noms des étiquettes (fonction externe)
        donnees.close()  # Fermeture du fichier pour libérer les ressources

    def __len__(self):
        """
        Retourne la taille du jeu de données.
        """
        return self.signaux.shape[0]

    def __getitem__(self, index):
        """
        Retourne un échantillon à l'indice donné.

        Paramètre :
        - i : indice de l'échantillon à récupérer.

        Retour :
        - Un tuple contenant (signal, SNR, étiquette_id) pour l'échantillon spécifié.
        """
        return (self.signaux[index], self.snr[index], self.etiquettes_id[index])
    

class SimpleModelTrainer(object):
    """
    Classe pour entraîner et tester un modèle de réseau neuronal.
    Elle prend en charge l'entraînement, l'évaluation sur un jeu de validation, 
    l'arrêt anticipé basé sur la performance du modèle, et la sauvegarde/chargement du modèle.
    """
    
    def __init__(self, model, verbose=True, device="cpu"):
        self.model = model
        self.verbose = verbose  # Permet d'afficher des informations détaillées pendant l'entraînement
        self.device = device  # Spécifie l'appareil (CPU ou GPU) pour l'entraînement
        self.train_loss = []
        self.accuracy_test = []
        self.test_loss = []

    def fit(self, n_epochs=100, path_to_data=False, dataloader=None, batch_size=32, lr=1e-5, valid_loader=None,
            critic_test=5, criterion=nn.NLLLoss(), model_path="model.pth", patience=5):
        """
        Entraîne le modèle sur les données d'entraînement, en effectuant des tests réguliers sur les données de validation.
        Si la performance du modèle ne s'améliore pas pendant 'patience' epochs, l'entraînement s'arrête.
        """
        self.critic_test = critic_test
        if not path_to_data and dataloader is None:
            raise ValueError("Please insert a dataloader or a path to the dataset")
        if dataloader is None:
            dataloader = DataLoader(MyDataset(path_to_data), shuffle=True, batch_size=batch_size)
        
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Optimiseur Adam
        initial_loss = 0
        test_loss = 0
        count_patience = 0

        with tqdm(total=n_epochs, desc=f"Epoch 0/{n_epochs} - Train Loss: {initial_loss:.4f} - Test Loss: {test_loss:.4f}", leave=True) as epoch_bar:
            for epoch in range(n_epochs):
                epoch_loss = 0  # Calcul de la perte pour l'epoch
                self.model.train()          
                for signals, _, labels in dataloader:
                    signals, labels = signals.to(self.device), labels.to(self.device).long()

                    optimizer.zero_grad()  # Remise à zéro des gradients avant chaque itération

                    outputs = self.model(signals)
                    loss = criterion(outputs, labels)
                    loss.backward()  # Calcul des gradients

                    optimizer.step()  # Mise à jour des poids du modèle
                    epoch_loss += loss.item()  # Accumulation de la perte de l'epoch
                
                avg_loss = epoch_loss / len(dataloader)
                epoch_bar.set_description(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_loss:.4f} - Test Loss: {test_loss:.4f}")
                epoch_bar.update(1)
                self.train_loss.append(avg_loss)

                # Test du modèle sur les données de validation à intervalles réguliers
                if valid_loader is not None and epoch % critic_test == 1:
                    test_labels, test_preds, t_loss = self._test(valid_loader, training=True, criterion=criterion)
                    test_loss = t_loss / len(valid_loader)

                    if test_loss < min(self.test_loss) if self.test_loss else np.inf:
                        self._save_model(model_path, verbose=False)  # Sauvegarde du modèle si amélioration
                        count_patience = 0 
                    count_patience += 1
                    self.test_loss.append(test_loss)
                    self.accuracy_test.append(accuracy_score(test_labels, test_preds))

                # Arrêt anticipé si la perte de validation ne s'améliore pas
                if count_patience > patience:
                    break 
        
        self._load_model(model_path, return_model=False, verbose=False)


## Same model as in the article

class PrototypeEncoder(nn.Module):
    def __init__(self, input_channels=2, output_dim=64):
        super(PrototypeEncoder, self).__init__()
        
        # Define 4 convolutional blocks for 1D signals
        self.block1 = self._conv_block(input_channels, 32)
        self.block2 = self._conv_block(32, 64)
        self.block3 = self._conv_block(64, 128)
        self.block4 = self._conv_block(128, output_dim)
        
    def _conv_block(self, in_channels, out_channels):
        """Helper function to define a single convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        """
        Forward pass for the embedding function.
        :param x: Input tensor of shape (batch_size, input_channels, signal_length)
        :return: Embeddings of shape (batch_size, output_dim, reduced_length)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # Optionally flatten if required for downstream tasks
        x = torch.flatten(x, start_dim=1)  # Shape: (batch_size, output_dim * reduced_length)
        return x