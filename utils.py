import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, default='EEG', help='EEG|DeepConv')
    parser.add_argument('-a', '--activation', type=str, default='ReLU', help='ReLU|LeakyReLU|ELU')
    args = parser.parse_args()

    print(args)

    return args


def read_bci_data() -> tuple[DataLoader, DataLoader]:
    S4b_train = np.load('dataset/S4b_train.npz')
    X11b_train = np.load('dataset/X11b_train.npz')
    S4b_test = np.load('dataset/S4b_test.npz')
    X11b_test = np.load('dataset/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.long)
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_label, dtype=torch.long)

    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')
    print(f'x_test: {x_test.shape}, y_test: {y_test.shape}')

    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=64, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=8, shuffle=True)

    return (train_loader, test_loader)


def show(history: dict[str, list]) -> None:
    plt.title('Train')

    plt.plot(history['loss'], label='loss')
    plt.plot(history['accuracy'], label='accuracy')

    plt.legend()
    plt.show()

    if ('val_loss' in history) and ('val_accuracy' in history):
        plt.title('Valid')

        plt.plot(history['val_loss'], label='loss')
        plt.plot(history['val_accuracy'], label='accuracy')

        plt.legend()
        plt.show()
