import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ustawienie globalnie większej czcionki
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})

# Wczytanie danych
file_path = r"C:\mgr\teeth_segment\RESULTS\segformer_loss_function.xlsx"
df = pd.read_excel(file_path)
# Konwersja przecinków na kropki i typ float
df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

# Przygotowanie danych - grupowanie po metryce
# Wybieramy kolumny metryk (bez "Inference Time (ms)")
metrics = df.columns[1:]

# Transpozycja: metryki jako wiersze, funkcje straty jako kolumny
df_plot = df.set_index('loss_function')[metrics].T

# Rysowanie wykresu słupkowego
fig, ax = plt.subplots(figsize=(16, 8))
df_plot.plot(kind='bar', ax=ax)
ax.set_xlabel("Metryka jakości oceny")
ax.set_ylabel("Wartość")
ax.set_title("Porównanie funkcji straty SegFormer wg metryk jakości")
ax.set_xticklabels(df_plot.index, rotation=45)
ax.set_ylim(0, 1)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(title="Funkcja straty", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
