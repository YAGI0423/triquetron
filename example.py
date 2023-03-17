import torch
from torch import nn
from torch.nn import functional as F

from triquetron import Linear, OneWeightLinear

from torch.utils.data import DataLoader
from logicGateDataset.datasets import AndGate, OrGate, XorGate, NotGate

from matplotlib import pyplot as plt

class TriModel(nn.Module):
    def __init__(self) -> None:
        super(TriModel, self).__init__()

        # self.layer1 = OneWeightLinear(2, 2)
        self.layer2 = OneWeightLinear(2, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # out = F.relu(self.layer1(input))
        return torch.sigmoid(self.layer2(input))


if __name__ == '__main__':
    dataset = XorGate(dataset_size=20000)
    dataLoader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TriModel()

    cri = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.003)

    losses = list()
    for x, y in dataLoader:
        y_hat = model(x)
        loss = cri(y_hat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss)
        losses.append(loss.item())
        
    plt.plot(losses)
    plt.show()