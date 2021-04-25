import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import parse, read_bci_data, show
from Net import EEGNet, DeepConvNet
from Model import Model


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    train_loader, test_loader = read_bci_data()

    net = None
    if args.net == 'EEG':
        net = EEGNet(args.activation)
    if args.net == 'DeepConv':
        net = DeepConvNet(args.activation)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    model = Model(net, criterion=criterion, optimizer=optimizer)
    model.summary((1, 1, 2, 750))

    history = model.train(train_loader, epochs=300, val_loader=test_loader)
    model.test(test_loader)

    with open(args.net + '_' + args.activation + '.txt', 'w') as txt_file:
        print(history['accuracy'], file=txt_file)
        print(history['val_accuracy'], file=txt_file)

    show(history)
