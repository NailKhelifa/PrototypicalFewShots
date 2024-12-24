import h5py
import torch


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, train_data_path):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(TrainDataset, self).__init__()

        # Chargement des données HDF5
        with h5py.File(train_data_path, 'r') as data:
            x_train = torch.tensor(data['signaux'][:])  # Convertir les signaux en tenseur PyTorch
            # Forme [30000, 2048, 2] que l'on réorganise en [30000, 2, 2048] : (batch, canal, résolution_temporelle)
            x_train = x_train.permute(0, 2, 1)  
            snr_train = torch.tensor(data['snr'][:])    
            y_train = torch.tensor(data['labels'][:])  # Convertir les labels en tenseur PyTorch

        self.x_train = x_train
        self.y_train = y_train
        self.snr_train = snr_train

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def __len__(self):
        return len(self.x_train)



