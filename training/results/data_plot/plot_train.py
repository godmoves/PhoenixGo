import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_tag(folder_name_list, tag_name, fig_id, ylim=None):
    file_name = "run_train-tag-{}.csv".format(tag_name)
    plt.figure(fig_id, figsize=(12, 8))
    for folder_name in folder_name_list:
        file_path = os.path.join("data/", folder_name, file_name)
        data = pd.read_csv(file_path)
        step = data["Step"] / 1000
        value = data["Value"]
        plt.plot(step, value, label=folder_name)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel("steps (k)")
    plt.title(tag_name)
    plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folders', nargs='*', default=[],
                        help='folders that contain training data')
    args = parser.parse_args()

    tag_name_list = ["Accuracy", "MSE Loss", "Total Loss"]
    for fig_id, tag_name in enumerate(tag_name_list):
        plot_tag(args.folders, tag_name, fig_id)
    plt.show()
