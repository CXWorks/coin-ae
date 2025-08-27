import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pickle
import torch.nn.functional as F
import torch

# Load first dataset
with open('data.pkl.10000', 'rb') as fp:
    probabilities, ground_truth = pickle.load(fp)

# Compute PR curve and AUPRC for the first model
precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
auprc = auc(recall, precision)

# Create a new figure with reduced height (e.g. 6" wide × 2" tall)
plt.figure(figsize=(6, 3))

plt.plot(recall, precision,
         label=f'Llama3.2-3B (AUPRC = {auprc:.3f})')

# Load LLama3.2-3B data
with open('qwen_data.pkl', 'rb') as fp:
    logits, ground_truth, _ = pickle.load(fp)
logits = np.array(logits)
probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
auprc = auc(recall, precision)

plt.plot(recall, precision,
         label=f'QWen2.5-1.5B (AUPRC = {auprc:.3f})',
         linestyle='--')

# Load QWen3-4B data
# with open('qwen3_data.pkl', 'rb') as fp:
#     logits, ground_truth = pickle.load(fp)
# logits = np.array(logits)
# probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
# precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
# auprc = auc(recall, precision)
#
# plt.plot(recall, precision,
#          label=f'QWen3-4B (AUPRC = {auprc:.3f})',
#          linestyle='-.')
#
# # Load Llama3.1-8B data (reused file in original example)
# with open('llama31_data.pkl', 'rb') as fp:
#     logits, ground_truth = pickle.load(fp)
# logits = np.array(logits)
# probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
# precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
# auprc = auc(recall, precision)
#
# plt.plot(recall, precision,
#          label=f'Llama3.1-8B (AUPRC = {auprc:.3f})',
#          linestyle=':')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to PDF. The figure will be short in height because figsize=(6, 2)
plt.savefig("pr-qwen-curve.pdf", format="pdf", bbox_inches="tight")
