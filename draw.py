import glob
import matplotlib.pyplot as plt


def show(net: str) -> dict[str, float]:
    max_accuracy = {}

    plt.figure(figsize=(12, 8))

    history_files = glob.glob(f'scores/{net}_*.txt')
    print(history_files)

    for history_file in history_files:
        with open(history_file, 'r') as txt_file:

            train_acc = list(map(lambda x: float(x), txt_file.readline().strip()[1: -1].split(', ')))
            test_acc = list(map(lambda x: float(x), txt_file.readline().strip()[1: -1].split(', ')))

            key = history_file.split('.')[0].split('_')[-1]
            max_accuracy[key] = max(test_acc)

            plt.plot(train_acc, label=f'{key} Train')
            plt.plot(test_acc, label=f'{key} Test')

    plt.legend()
    plt.show()

    return max_accuracy


if __name__ == '__main__':
    max_accuracy = {}

    max_accuracy['EEG'] = show('EEG')
    max_accuracy['DeepConv'] = show('DeepConv')

    print()
    print(f'\t{" " * 15} | {"ReLU":^11} | {"Leaky ReLU":^11} | {"ELU":^11} |')
    print(f'\t{"-" * 60}')
    print(f'\t{"EEGNet":<15} | {max_accuracy["EEG"]["ReLU"]:^11.3f} | {max_accuracy["EEG"]["LeakyReLU"]:^11.3f} | {max_accuracy["EEG"]["ELU"]:^11.3f} |')
    print(f'\t{"DeepConvNet":<15} | {max_accuracy["DeepConv"]["ReLU"]:^11.3f} | {max_accuracy["DeepConv"]["LeakyReLU"]:^11.3f} | {max_accuracy["DeepConv"]["ELU"]:^11.3f} |')
    print()
