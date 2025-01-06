import os
import numpy as np
import torch
from tqdm import trange
import h5py
from torch.utils.data import Dataset

class SignalNShotTrainDataset(Dataset):
    def __init__(self, train_data_path, batch_size=100, classes_per_set=10, samples_per_class=1, matching=False,
                 p_data=1):
        """
        Dataset combinant les fonctionnalités de TrainDataset et OmniglotNShotDataset.
        Args:
        - train_data_path: Chemin vers le fichier HDF5 contenant les données.
        - batch_size: Taille des batchs.
        - classes_per_set: Nombre de classes par ensemble pour le N-shot learning.
        - samples_per_class: Nombre d'échantillons par classe.
        """
        super(SignalNShotTrainDataset, self).__init__()

        # Chargement des données HDF5
        with h5py.File(train_data_path, 'r') as data:
            x_train = torch.tensor(data['signaux'][:])  # Forme initiale [30000, 2048, 2]
            x_train = x_train.permute(0, 2, 1)  # Réorganiser en [30000, 2, 2048]
            snr_train = torch.tensor(data['snr'][:])    
            y_train = torch.tensor(data['labels'][:])  # Labels

        if not (0 < p_data <= 1.0):
            raise ValueError("p_data doit être entre 0 et 1.")

        total_samples = x_train.size(0)
        num_samples = int(total_samples * p_data)
        indices = torch.randperm(total_samples)[:num_samples]

        x_train = x_train[indices]
        snr_train = snr_train[indices]
        y_train = y_train[indices]

        self.x_train = x_train
        self.y_train = y_train
        self.snr_train = snr_train

        # Organisation pour le N-shot learning
        self.batch_size = batch_size
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        self.n_classes = len(torch.unique(self.y_train))
        if matching:
            self.normalization()
            # Organisation des datasets (pour compatibilité avec N-shot learning)
            self.indexes = {"train": 0}
            self.datasets_cache = {"train": self.load_data_cache()}  # Cache des batchs

    def normalization(self):
        """
        Normalise les signaux pour avoir une moyenne de 0 et un écart-type de 1.
        """
        self.mean = torch.mean(self.x_train)
        self.std = torch.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std

    def load_data_cache(self):
        """
        Prépare des batchs pour le N-shot learning.
        """
        n_samples = self.samples_per_class * self.classes_per_set
        data_cache = []

        for _ in trange(300):
            support_set_x = torch.zeros((self.batch_size, n_samples, 2, 2048))
            support_set_y = torch.zeros((self.batch_size, n_samples))
            target_x = torch.zeros((self.batch_size, self.samples_per_class, 2, 2048))
            target_y = torch.zeros((self.batch_size, self.samples_per_class))

            for i in range(self.batch_size):
                pinds = np.random.permutation(n_samples)
                classes = np.random.choice(self.n_classes, self.classes_per_set, replace=False)
                x_hat_class = np.random.choice(classes, self.samples_per_class, replace=True)
                pinds_test = np.random.permutation(self.samples_per_class)

                ind = 0
                ind_test = 0
                for j, cur_class in enumerate(classes):
                    class_inds = (self.y_train == cur_class).nonzero(as_tuple=True)[0]
                    if cur_class in x_hat_class:
                        n_test_samples = torch.sum(torch.tensor(cur_class) == torch.tensor(x_hat_class)).item()
                        example_inds = np.random.choice(class_inds.numpy(), self.samples_per_class + n_test_samples, replace=False)
                    else:
                        example_inds = np.random.choice(class_inds.numpy(), self.samples_per_class, replace=False)

                    # Meta-training
                    for eind in example_inds[:self.samples_per_class]:
                        support_set_x[i, pinds[ind], :, :] = self.x_train[eind]
                        support_set_y[i, pinds[ind]] = j
                        ind += 1

                    # Meta-test
                    for eind in example_inds[self.samples_per_class:]:
                        target_x[i, pinds_test[ind_test], :, :] = self.x_train[eind]
                        target_y[i, pinds_test[ind_test]] = j
                        ind_test += 1

            data_cache.append([support_set_x, support_set_y, target_x, target_y])

        return data_cache

    def __get_batch(self):
        """
        Récupère le prochain batch pour le N-shot learning.
        """
        if self.indexes["train"] >= len(self.datasets_cache["train"]):
            self.indexes["train"] = 0
            self.datasets_cache["train"] = self.load_data_cache()

        next_batch = self.datasets_cache["train"][self.indexes["train"]]
        self.indexes["train"] += 1
        return next_batch

    def get_batch(self):
        """
        Récupère un batch pour l'entraînement N-shot.
        """
        return self.__get_batch()

    def __getitem__(self, idx):
        """
        Compatible avec le DataLoader de PyTorch : renvoie un signal et son label.
        """
        return self.x_train[idx], self.y_train[idx]

    def __len__(self):
        """
        Retourne la taille du dataset.
        """
        return len(self.x_train)
