"""
This code is used to draw vocabulary overlap between training and different types of test sets
"""
import json
import matplotlib.pyplot as plt
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

with open("averaged_logs.json", 'r') as f:
    logs = json.load(f)

metrics = logs['measure_implicature_original'].keys()
metrics_p = ['ROUGE-L' if k == 'rougeL' else k for k in metrics]
original = [logs['measure_implicature_original'][k] for k in metrics]
negation = [logs['measure_implicature_negation'][k] for k in metrics]
implicature = [logs['measure_implicature_implicature'][k] for k in metrics]

barWidth = 0.25
bar_orig = np.arange(len(original))
bar_neg = [x + barWidth for x in bar_orig]
bar_imp = [x + barWidth for x in bar_neg]

fig, ax = plt.subplots(figsize=(12,8))
# for k in metrics:
#     ax.plot(['original', 'negation', 'implicature'], [original[k], negation[k], implicature[k]], lw=2, label=k)
ax.bar(bar_orig, original, color='#FCC074', width=barWidth,
       label='original')
ax.bar(bar_neg, negation, color='#A8FCA7', width=barWidth,
       label='negation')
ax.bar(bar_imp, implicature, color='#FE91CA', width=barWidth,
       label='implicature')

# ax.set_xticklabels(['original', 'negation', 'implicature'])
ax.legend()
ax.grid()
ax.set_xticks([x + barWidth for x in range(len(original))], metrics_p)
plt.savefig('vocab.png', bbox_inches='tight', pad_inches=0, dpi=100)
plt.savefig('vocab.pdf', bbox_inches='tight', pad_inches=0, dpi=400)