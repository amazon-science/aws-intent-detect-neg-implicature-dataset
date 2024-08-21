import os, json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

TINY_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=TINY_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

model_name="results/pretrain_data_v3_instructor_base_252744_proc_disable_hard_pos_7_hard_neg_2"
baseline_name="../evaluate_results/averaged_logs.json"
encoder_name="instructor-base"

baselines = {}
with open(baseline_name, 'r') as f:
    data = json.load(f)
    for measure in data:
        baselines[measure] = data[measure][encoder_name]

steps = []
logs = defaultdict(list)
for checkpoint in os.listdir(model_name):
    if not checkpoint.startswith("checkpoint"):
        continue
    checkpoint_logs = os.path.join(model_name, checkpoint, "averaged_logs.json")
    step = int(checkpoint.split("-")[-1])
    with open(checkpoint_logs, 'r') as f:
        steps.append(step)
        log = json.load(f)
        for measure in log:
            logs[measure].append(log[measure])

order = np.argsort(steps)
steps = [steps[i] for i in order]
for measure in logs:
    logs[measure] = [logs[measure][i] for i in order]

os.makedirs(os.path.join(model_name, "figures"), exist_ok=True)
for measure in logs:
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(steps, logs[measure], label="Ours")
    ax.axhline(y=baselines[measure], color='r', ls='-', label="baseline")
    ax.set_xlabel("Training Steps")
    ax.legend()
    plt.savefig(os.path.join(model_name, "figures", f"{measure.replace(' ', '')}_vs_epoch.png"), bbox_inches='tight', pad_inches=0, dpi=100)
