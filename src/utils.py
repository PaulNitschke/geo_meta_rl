import torch as th
    
class DenseNN(th.nn.Module):
    """A fully connected neural network with arbitrary layer sizes and ReLU activations."""
    def __init__(self, layer_sizes: list[int]):
        """
        Args:
            layer_sizes (list of int): List of layer sizes, including input and output dimensions.
                                       Example: [2, 64, 128, 2]
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(th.nn.Linear(in_dim, out_dim))
            layers.append(th.nn.ReLU()) 
        layers.pop()
        self.net = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)