import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
file_path = r"C:\mgr\teeth_segment\RESULTS\unet_loss_function.xlsx"
df = pd.read_excel(file_path)
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Przygotowanie danych
metrics = df.columns[1:-1]  # bez "Inference Time (ms)"
num_metrics = len(metrics)
bar_width = 0.12

# Podział na 2 grupy po 3 funkcje straty
groups = [df.iloc[:3], df.iloc[3:]]
fig, axes = plt.subplots(nrows=2, figsize=(14, 10), sharey=True)

for ax, group in zip(axes, groups):
    x = np.arange(len(group))  # 0,1,2 dla 3 funkcji
    for i, metric in enumerate(metrics):
        values = group[metric]
        ax.bar(x + i * bar_width, values, width=bar_width, label=metric if ax == axes[0] else "")
    
    ax.set_xticks(x + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(group['loss_function'], rotation=45)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wartość")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

axes[0].set_title("Porównanie metryk – Funkcje straty (grupa 1)")
axes[1].set_title("Porównanie metryk – Funkcje straty (grupa 2)")
axes[0].legend(loc='upper right')
plt.tight_layout()
plt.show()
