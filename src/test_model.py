import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .proto_encoder import PrototypeEncoder
import h5py
import os
from .utils import get_n_per_classes


def compute_prototypes(encoder, x_enroll, y_enroll):
    """
    Enrôle les classes et calcule leurs prototypes.

    :param encoder: Modèle PrototypeEncoder déjà entraîné.
    :param x_enroll: Tenseur contenant les signaux d'enrôlement de taille (N, 2, 2048).
    :param y_enroll: Tenseur contenant les labels des signaux d'enrôlement de taille (N,).
    :return: Dictionnaire contenant les prototypes {label: prototype_tensor}.
    """
    # Activer le mode évaluation (pas d'entraînement, désactive dropout si utilisé)
    encoder.eval()

    # Calcul des embeddings pour les signaux d'enrôlement
    with torch.no_grad():  # Pas de calcul des gradients pour gagner en mémoire et en vitesse
        embeddings = encoder(x_enroll)  # Shape: (N, output_dim)

    # Trouver les classes uniques
    unique_classes = torch.unique(y_enroll)

    # Calculer les prototypes
    prototypes = {}
    for cls in unique_classes:
        # Sélectionner les embeddings appartenant à la classe cls
        class_embeddings = embeddings[y_enroll == cls]  # Shape: (num_samples_in_class, output_dim)

        # Calculer le prototype comme la moyenne des embeddings
        prototype = class_embeddings.mean(dim=0)  # Shape: (output_dim,)

        # Ajouter le prototype au dictionnaire
        prototypes[cls.item()] = prototype

    return prototypes

def compute_accuracy(encoder, prototypes, x_test, y_test, distance_metric="euclidean",
                     logs=False):
    """
    Calcule l'accuracy des prédictions sur un ensemble de test.

    :param encoder: Modèle PrototypeEncoder déjà entraîné.
    :param prototypes: Dictionnaire contenant les prototypes {label: prototype_tensor}.
    :param x_test: Tenseur des signaux de test de taille (N_test, 2, 2048).
    :param y_test: Tenseur des labels de test de taille (N_test,).
    :param distance_metric: Métrique de distance à utiliser ("euclidean" ou "cosine").
    :return: Accuracy (valeur entre 0 et 1).
    """
    # Activer le mode évaluation
    device = x_test.device
    encoder.eval()
    encoder.to(device)
    # Obtenir les embeddings des données de test
    with torch.no_grad():
        test_embeddings = encoder(x_test)  # Shape: (N_test, output_dim)
    
    # Convertir les prototypes en un tenseur pour faciliter les calculs
    prototype_labels = torch.tensor(list(prototypes.keys())).to(device)  # Labels des prototypes
    prototype_vectors = torch.stack(list(prototypes.values())).to(device)  # Shape: (num_classes, output_dim)
    
    # Calculer les distances entre chaque embedding de test et chaque prototype
    if distance_metric == "euclidean":
        # Distance euclidienne : ||f(x) - c_k||^2
        distances = torch.cdist(test_embeddings, prototype_vectors)  # Shape: (N_test, num_classes)
    elif distance_metric == "cosine":
        # Distance cosinus : 1 - cos(f(x), c_k)
        distances = 1 - F.cosine_similarity(
            test_embeddings.unsqueeze(1),  # Shape: (N_test, 1, output_dim)
            prototype_vectors.unsqueeze(0),  # Shape: (1, num_classes, output_dim)
            dim=-1
        )  # Shape: (N_test, num_classes)
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")
    
    # Prédictions : choisir le prototype le plus proche pour chaque point de test
    predicted_labels = prototype_labels[torch.argmin(distances, dim=1)]  # Shape: (N_test,)

    if logs:
        return (predicted_labels, y_test)

    accuracy = (predicted_labels == y_test).float().mean().item()

    return accuracy


def load_data(task="enroll", data_dir="../data"):
    if task == "enroll":
        data_dir += "/enroll.hdf5"
    elif task == "test":
        data_dir += '/test_fewshot.hdf5'
    elif task == "train":
        data_dir += "/train.hdf5"
    # Chargement des données HDF5
    with h5py.File(data_dir, 'r') as data:
        x = torch.tensor(data['signaux'][:])  # Convertir les signaux en tenseur PyTorch
        # Forme [N_samples, 2048, 2] que l'on réorganise en [N_samples, 2, 2048] : (batch, canal, résolution_temporelle)
        x = x.permute(0, 2, 1)  
        snr = torch.tensor(data['snr'][:])    
        y = torch.tensor(data['labels'][:])  # Convertir les labels en tenseur PyTorch

    return x, y, snr


def test_acc(model_path, n_shot=500, data_dir="../data"):
    '''
    model_path : répertoire auquel est sauvegardé le modèle à tester
    n_shot : nombre d'échantillons à considérer pour enroller les nouvelles classes
    '''
    model = PrototypeEncoder()

    state_dict_path_20 = model_path  # Path to the saved state_dict file

    state_dict_20 = torch.load(state_dict_path_20, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict_20)
    # Set the model to evaluation mode (if needed)
    model.eval()

    x_enroll, y_enroll, _ = load_data(task="enroll", data_dir=data_dir)
    
    # on ne retient que les n_shot-premières données pour enroller les classes
    sampled_x_enroll, sampled_y_enroll = x_enroll[:n_shot], y_enroll[:n_shot]
    x_test, y_test, _ = load_data(task="test")

    # compute the prototypes
    prototypes = compute_prototypes(model, sampled_x_enroll, sampled_y_enroll)

    # compute the accuracies on the test
    accuracy = compute_accuracy(model, prototypes, x_test, y_test)

    return accuracy

def test_n_shot(model, n=5, device="cuda:0", return_logs=False, data_dir="../data"):
    """
    Si return_logs==True: retourne les preds et labels (dans un tuple)
    sinon retourne l'accuracy
    """
    x_enroll, y_enroll, _ = load_data(task="enroll", data_dir=data_dir)
    x_enroll, y_enroll = x_enroll.to(device), y_enroll.to(device)
    (x_n, y_n), idxs = get_n_per_classes(x_enroll, y_enroll, n=n)

    model.to(device)
    prototype = compute_prototypes(model, x_n, y_n)

    x_test, y_test, _ = load_data(task="test", data_dir=data_dir)
    x_test, y_test = x_test.to(device), y_test.to(device)

    out = compute_accuracy(model, prototype, x_test, y_test, logs=return_logs)

    return out
