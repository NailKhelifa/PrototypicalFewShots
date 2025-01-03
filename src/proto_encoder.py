import torch.nn as nn

## Same model as in the article adapter in our setting with signals
'''
Our embedding architecture mirrors that used by Vinyals et al. [29] and is composed of four
convolutional blocks. Each block comprises a 64-filter 3 × 3 convolution, batch normalization layer
[10], a ReLU nonlinearity and a 2 × 2 max-pooling layer. When applied to the 28 × 28 Omniglot
images this architecture results in a 64-dimensional output space.
'''
class PrototypeEncoder(nn.Module):
    def __init__(self, input_channels=2, output_dim=64, hidden_dim=64):
        super(PrototypeEncoder, self).__init__()
        
        # Define 4 convolutional blocks for 1D signals
        self.block1 = self._conv_block(input_channels, hidden_dim)
        self.block2 = self._conv_block(hidden_dim, hidden_dim)
        self.block3 = self._conv_block(hidden_dim, hidden_dim)
        self.block4 = self._conv_block(hidden_dim, output_dim)
        
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
        return x.view(x.size(0), -1)