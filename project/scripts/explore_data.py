import pandas as pd

# Scegli uno qualsiasi dei file dal tuo training set
file_path = 'training_dataset_completo.tsv' 

# I file sono 'tab-separated', quindi usiamo sep='\t'
df = pd.read_csv(file_path, sep='\t')

# Stampa le prime 5 righe per vedere la struttura
print("Prime 5 righe del dataset:")
print(df.head())

# Stampa i nomi di tutte le colonne
print("\nNomi delle colonne:")
print(df.columns)

# Stampa quante volte compare ogni tipo di interazione in questo file
print("\nConteggio delle interazioni nel file:")
print(df['Interaction'].value_counts())