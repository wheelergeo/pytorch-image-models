import os
import json
import re
import csv
import matplotlib.pyplot as plt


from typing import List, Tuple


def plot_performance(perf_dir: str = "./workdir/performance",
                     file_regex: str = r"performance_tome_r-(\d+)\.json",
                     save_path: str = "./workdir/performance/performance.png"):
    pattern = re.compile(file_regex)
    indices, flops, accuracy, throughput = [], [], [], []

    assert os.path.exists(perf_dir), f"Performance directory {perf_dir} does not exist."

    for fname in os.listdir(perf_dir):
        match = pattern.match(fname)
        if match:
            i = int(match.group(1))
            with open(os.path.join(perf_dir, fname), "r") as f:
                data = json.load(f)
            indices.append(i)
            # 去掉单位，只保留数字
            flops.append(float(data["flops"].replace("G", "")))
            accuracy.append(float(data["accuracy"]))
            throughput.append(float(data["throughput"]))

    sorted_data = sorted(zip(indices, flops, accuracy, throughput))
    indices, flops, accuracy, throughput = map(list, zip(*sorted_data))

    plt.figure(figsize=(10, 6))
    # gca() 获取当前的坐标轴对象
    ax1 = plt.gca()
    # 创建共享x轴的第二个y轴
    ax2 = ax1.twinx()
    ax1.plot(indices, flops, 'b--', label="FLOPs (g)")
    ax1.plot(indices, throughput, 'b-', label="Throughput (im/s)")
    ax2.plot(indices, accuracy, 'r-', label="Accuracy (%)")

    ax1.set_xlabel("r")
    ax1.set_ylabel("FLOPs / Throughput", color='b')
    ax2.set_ylabel("Accuracy", color='r')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax2.set_ylim(55.0, 90.0)
    plt.title("Token Merging Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)


def plot_performance2(perf_dirs: List[Tuple[str, str]] = [
    ('deit-tiny-patch16-224-fb-in1k', 'vit-tiny-patch16-224-augreg-in21k-ft-in1k'),
    ('deit-small-patch16-224-fb-in1k', 'vit-small-patch16-224-augreg-in1k'),
    ('deit-base-patch16-224-fb-in1k', 'vit-base-patch16-224-augreg-in1k'),
    ('deit3-large-patch16-224-fb-in22k-ft-in1k', 'vit-large-patch16-224-augreg-in21k-ft-in1k')],
                     file_regex: str = r"performance_tome_r-(\d+)\.json",
                     save_path: str = "./workdir/performance.png"):
    pattern = re.compile(file_regex)
    list_len = len(perf_dirs)
    tome_r = [{'origin': [], 'augreg': []} for _ in range(list_len)]
    acc = [{'origin': [], 'augreg': []} for _ in range(list_len)]

    for i, tp in enumerate(perf_dirs):
        origin = './workdir/' + tp[0] + '.perf'
        assert os.path.exists(origin), f"Performance directory {origin} does not exist."
        for fname in os.listdir(origin):
            match = pattern.match(fname)
            if match:
                r = int(match.group(1))
                with open(os.path.join(origin, fname), "r") as f:
                    data = json.load(f)
                tome_r[i]['origin'].append(r)
                acc[i]['origin'].append(float(data["accuracy"]))
        sorted_data = sorted(zip(tome_r[i]['origin'], acc[i]['origin']))
        tome_r[i]['origin'], acc[i]['origin'] = map(list, zip(*sorted_data))

        augreg = './workdir/' + tp[1] + '.perf'
        assert os.path.exists(augreg), f"Performance directory {augreg} does not exist."
        for fname in os.listdir(augreg):
            match = pattern.match(fname)
            if match:
                r = int(match.group(1))
                with open(os.path.join(augreg, fname), "r") as f:
                    data = json.load(f)
                tome_r[i]['augreg'].append(r)
                acc[i]['augreg'].append(float(data["accuracy"]))
        sorted_data = sorted(zip(tome_r[i]['augreg'], acc[i]['augreg']))
        tome_r[i]['augreg'], acc[i]['augreg'] = map(list, zip(*sorted_data))

    color_list = ["black", "blue", "red", "orange", "purple", "cyan", "magenta", "green"]
    plt.figure(figsize=(10, 6))
    # gca() 获取当前的坐标轴对象
    ax1 = plt.gca()
    for i in range(list_len):
        ax1.plot(tome_r[i]['origin'], acc[i]['origin'], color=color_list[i], linestyle='-', label=f"{perf_dirs[i][0]}")
        ax1.plot(tome_r[i]['augreg'], acc[i]['augreg'], color=color_list[i], linestyle='-.', label=f"{perf_dirs[i][1]}")

    ax1.set_xlabel("r")
    ax1.set_ylabel("Accuracy", color='black')
    ax1.legend()
    ax1.set_ylim(30.0, 90.0)
    plt.title("ToMe Accuracy Comparision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)


def plot_from_csv(csv_path: str = "./checkpoints/my-vit-small-patch16-224/20250806-181915-vit_small_patch16_224/summary.csv",
                  save_path: str = "./checkpoints/my-vit-small-patch16-224/20250806-181915-vit_small_patch16_224/summary.png"):
    epochs = []
    train_loss = []
    eval_loss = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            eval_loss.append(float(row['eval_loss']))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, eval_loss, label='Eval Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Eval Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
