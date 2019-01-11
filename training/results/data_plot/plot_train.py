import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os


def prepare_data(folder_name, file_name, fig_idx):
    path = os.path.join(folder_name, file_name)
    data = pd.read_csv(path)
    step = data["Step"]
    value = data["Value"]

    offset = step[0]
    if offset != 0:
        # step = [s - offset for s in step]
        step = [s / 1000 for s in step]

    plt.figure(fig_idx, figsize=(12, 8))
    if fig_idx == 1:
        plt.title("accuracy")
        plt.ylim(0.35, 0.55)
    elif fig_idx == 2:
        plt.title("mse loss")
        plt.ylim(0.155, 0.20)
    else:
        plt.title("total loss")
        plt.ylim(2.9, 4)

    plt.xlabel("steps (k)")
    plt.plot(step, value, label=folder_name[7:])


def plot_train_loop(folder_name, plot_test=False):
    prepare_data(folder_name, "run_train-tag-Accuracy.csv", 1)
    prepare_data(folder_name, "run_train-tag-MSE Loss.csv", 2)
    prepare_data(folder_name, "run_train-tag-Total Loss.csv", 3)
    if plot_test:
        prepare_data(folder_name, "run_test-tag-Accuracy.csv", 1)
        prepare_data(folder_name, "run_test-tag-MSE Loss.csv", 2)
        prepare_data(folder_name, "run_test-tag-Total Loss.csv", 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', nargs='*', default=[],
                        help='folders that contain training data')
    parser.add_argument('-p', '--plot_test', action='store_true', default=False,
                        help='plot test data at the same time')
    parser.add_argument('-n', '--name', help='figure name')
    args = parser.parse_args()

    for f in args.folder:
        f = "data/" + f
        print(f)
        plot_train_loop(f, args.plot_test)
    for idx in range(3):
        plt.figure(idx + 1).legend()
        plt.figure(idx + 1).savefig("./image/" + args.name + str(idx), dpi=300)
