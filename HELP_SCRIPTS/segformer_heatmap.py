import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych z pliku Excel
file_path = r'C:\mgr\teeth_segment\RESULTS\segformer_hiperparameters.xlsx'
df = pd.read_excel(file_path)

# Wybór interesujących metryk
metrics = ['Accuracy', 'Precision', 'Recall', 'Dice Score', 'F2 Score', 'Jaccard Index']
combinations = df['Nr kobinacji']
metrics_df = df[metrics]
metrics_df.index = combinations

# Tworzenie heatmapy
plt.figure(figsize=(12, 8))
sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={'label': 'Score'})
plt.title("Heatmapa metryk względem numeru kombinacji")
plt.xlabel("Metryka")
plt.ylabel("Numer kombinacji")
plt.tight_layout()
plt.show()
