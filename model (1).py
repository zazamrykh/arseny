import torch.nn as nn


class FFNN(nn.Module):
    """
    input_size  — размер входа (784 для 28×28)
    hidden_size — число нейронов в скрытом слое (Ns)
    output_size — число классов (K букв)
    activation  — передаточная функция ('sigmoid','tanh','relu','leaky_relu')
    """

    activations = {
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
    }

    def __init__(self, input_size, hidden_size, output_size, activation="relu"):
        super(FFNN, self).__init__()

        if activation not in self.activations:
            raise ValueError(f"Неизвестная активация: {activation}.")

        layers = []
        prev = input_size
        for h in hidden_size:
            layers.append(nn.Linear(prev, h))
            layers.append(self.activations[activation])
            prev = h
        layers.append(nn.Linear(prev, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
