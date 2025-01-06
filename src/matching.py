import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi-layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3-layer LSTM with 100 neurons each
        :param batch_size: The experiments batch size
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(
            input_size=self.vector_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        outputs, (hn, cn) = self.lstm(x)
        return outputs, hn, cn


class ConvLayer(nn.Module):
    """1D convolution layer with optional dropout."""

    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(ConvLayer, self).__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.1))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Classifier(nn.Module):
    def __init__(self, layer_size, num_channels=2, n_classes=0, use_dropout=False, signal_length=2048):
        super(Classifier, self).__init__()
        """
        Builds a CNN to produce embeddings for signal data
        :param layer_size: Number of filters in the convolutional layers
        :param num_channels: Number of input channels (e.g., 2 for your signals)
        :param n_classes: If >0, adds a final FC layer for classification
        :param use_dropout: Use dropout with p=0.1 in each Conv block
        :param signal_length: Temporal resolution of the input signals
        """
        self.layer1 = ConvLayer(num_channels, layer_size, use_dropout)
        self.layer2 = ConvLayer(layer_size, layer_size, use_dropout)
        self.layer3 = ConvLayer(layer_size, layer_size, use_dropout)
        self.layer4 = ConvLayer(layer_size, layer_size, use_dropout)

        final_length = signal_length // (2 * 2 * 2 * 2)  # After 4 max-pooling layers
        self.out_size = final_length * layer_size

        if n_classes > 0:
            self.use_classification = True
            self.fc = nn.Linear(self.out_size, n_classes)
            self.out_size = n_classes
        else:
            self.use_classification = False

    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = x.view(x.size(0), -1)  # Flatten
        if self.use_classification:
            x = self.fc(x)
        return x


class DistanceNetwork(nn.Module):
    def forward(self, support_set, input_signal):
        """
        Computes cosine similarity between support set and target signal embeddings.
        :param support_set: [sequence_length, batch_size, embedding_dim]
        :param input_signal: [batch_size, embedding_dim]
        :return: Similarities [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        support_set = support_set.permute(1,0,2)

        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_signal.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


class AttentionalClassify(nn.Module):
    def forward(self, similarities, support_set_y):
        """
        Computes probabilities over support set classes.
        :param similarities: Cosine similarities [batch_size, sequence_length]
        :param support_set_y: One-hot labels of support set [sequence_length, batch_size, num_classes]
        :return: Predicted probabilities [batch_size, num_classes]
        """
        softmax = nn.Softmax(dim=1)

        softmax_similarities = softmax(similarities)
        #support_set_y = support_set_y.permute(1, 0, 2)  # Nouvelle taille : (5, 9, 3)

        softmax_similarities = softmax_similarities.unsqueeze(1)  # Taille : (5, 1, 9)

        preds = softmax_similarities.bmm(support_set_y)  # Résultat : (5, 1, 3)

        preds = preds.squeeze(1) 
        return preds

class MatchingNetwork(nn.Module):
    def __init__(self, batch_size=100, num_channels=2, fce=False, num_classes_per_set=5, 
                 num_samples_per_class=1, layer_size=64, signal_length=2048):
        super(MatchingNetwork, self).__init__()
        """
        Matching Network for signal data.
        """
        self.batch_size = batch_size
        self.fce = False
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class

        # Embedding network
        self.g = Classifier(layer_size, num_channels, signal_length=signal_length)

        # Full context embeddings (optional)
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=batch_size, vector_dim=self.g.out_size)

        # Distance and classification
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()

    def forward(self, support_set, support_set_y, target_signal, target_label):
        """
        Forward pass for Matching Network.
        """
        #support_set = support_set.permute(1,0,2,3)

        encoded_support = torch.stack([self.g(support) for support in support_set])

        encoded_target = self.g(target_signal)

        if self.fce:
            encoded_support, _, _ = self.lstm(encoded_support)

        #encoded_support = encoded_support.reshape(self.num_classes_per_set, int(self.num_samples_per_class*self.batch_size), encoded_support.shape[-1])

        similarities = self.dn(encoded_support, encoded_target)
        preds = self.classify(similarities, support_set_y)

        loss = F.cross_entropy(preds, target_label)
        accuracy = (preds.argmax(dim=1) == target_label).float().mean()

        return accuracy, loss
    
    def predict(self, support_set, support_set_y, target_signal):

        encoded_support = torch.stack([self.g(support) for support in support_set])

        encoded_target = self.g(target_signal)

        if self.fce:
            encoded_support, _, _ = self.lstm(encoded_support)

        #encoded_support = encoded_support.reshape(self.num_classes_per_set, int(self.num_samples_per_class*self.batch_size), encoded_support.shape[-1])

        similarities = self.dn(encoded_support, encoded_target)
        preds = self.classify(similarities, support_set_y)

        return preds