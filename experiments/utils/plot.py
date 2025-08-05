import os
import json
import re
import matplotlib.pyplot as plt


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

    ax2.set_ylim(50.0, 85.0)
    plt.title("Token Merging Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)