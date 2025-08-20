import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, matthews_corrcoef

# 1. Carica il tuo dataset completo
df = pd.read_csv('training_dataset_completo.tsv', sep='\t')

# Rimuovi tutte le righe in cui la colonna 'Interaction' è vuota (NaN).
print(f"Dimensione dataset prima della pulizia: {df.shape}")
df.dropna(subset=['Interaction'], inplace=True)
print(f"Dimensione dataset dopo la pulizia: {df.shape}")

# Seleziona le feature e il target che vuoi usare
numerical_features = ['s_rsa', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_rsa', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']
categorical_features = ['s_resn', 't_resn', 's_ss8', 't_ss8']

# --- ECCO LA CORREZIONE ---
target_column = 'Interaction'  # Definiamo il nome della colonna che vogliamo predire
# -------------------------

# Estrai la colonna target (y) PRIMA di fare il one-hot encoding
y = df[target_column]

# Seleziona le colonne delle feature che saranno elaborate
X_features = df[numerical_features + categorical_features]

# A partire da Pandas >1.5, è meglio lavorare su una copia per evitare SettingWithCopyWarning
X_features = X_features.copy()

# Gestisci i NaN nelle feature numeriche
for col in numerical_features:
    X_features[col].fillna(X_features[col].mean(), inplace=True)
    
# Converti le feature categoriche in numeriche con One-Hot Encoding
X_processed = pd.get_dummies(X_features, columns=categorical_features, dummy_na=False)

print(f"Il numero di feature è passato da {len(X_features.columns)} a {len(X_processed.columns)} dopo One-Hot Encoding.")

X = X_processed

# 2. Dividi i dati: una parte per addestrare, una per testare
#    IMPORTANTE: Ora y è definita correttamente e può essere usata qui
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crea e addestra il modello
model = GaussianNB()
print("Inizio addestramento...")
model.fit(X_train, y_train)
print("Addestramento completato.")

# 4. Fai delle predizioni sul set di test
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 5. Valuta quanto è bravo il tuo modello
accuracy = accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)

print(f"\nAccuracy sul test set: {accuracy:.4f}")
print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

# 6. Stampa un esempio di predizione
print("\n--- Esempio di predizione ---")
# Usiamo i nomi delle colonne per vedere meglio l'esempio
example_features = X_test.iloc[0] 
predicted_class = predictions[0]
actual_class = y_test.iloc[0]

print(f"Predizione del modello: {predicted_class}")
print(f"Vera etichetta: {actual_class}")

class_order = model.classes_
probabilities_for_example = probabilities[0]

print("\n--- Dettaglio Punteggio (Probabilità) ---")
print(f"Corrispondenza Classe -> Punteggio per l'esempio sopra:")
for i in range(len(class_order)):
    print(f"- {class_order[i]}: {probabilities_for_example[i]:.4f}")
print("----------------------------------------")