import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
import time
from imblearn.under_sampling import RandomUnderSampler
import joblib

# --- Caricamento e Preparazione Dati (invariato) ---
df = pd.read_csv('/content/project/training_dataset_completo.tsv', sep='\t')
print(f"Dimensione dataset prima della pulizia: {df.shape}")
df.dropna(subset=['Interaction'], inplace=True)
print(f"Dimensione dataset dopo la pulizia: {df.shape}")

# Definizioni
numerical_features = ['s_rsa', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_rsa', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']
categorical_features = ['s_resn', 't_resn', 's_ss8', 't_ss8']
target_column = 'Interaction'

# Estrazione X e y
y = df[target_column]
X_features = df[numerical_features + categorical_features].copy()

# Pulizia e One-Hot Encoding
for col in numerical_features:
    mean_value = X_features[col].mean()
    X_features[col] = X_features[col].fillna(mean_value)
X = pd.get_dummies(X_features, columns=categorical_features, dummy_na=False)

model_columns = X.columns
joblib.dump(model_columns, '/content/project/model_columns.joblib')

print(f"Numero finale di feature: {X.shape[1]}")

# --- Divisione Dati (invariato) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nDistribuzione delle classi PRIMA del bilanciamento (nel training set):")
print(y_train.value_counts())

# --- 2. APPLICHIAMO IL SOTTOCAMPIONAMENTO (UNDERSAMPLING) ---
print("\nApplicazione di RandomUnderSampler...")
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print("\nDistribuzione delle classi DOPO il bilanciamento (nel nuovo training set):")
print(y_train_resampled.value_counts())

# --- Addestramento Modello ---
# NOTA: Ora non usiamo più class_weight='balanced', perché abbiamo già bilanciato i dati!
model = RandomForestClassifier(
    n_estimators=100,      
    random_state=42,       
    n_jobs=-1
)

print("\nInizio addestramento sul dataset bilanciato...")
start_time = time.time()
# IMPORTANTE: addestriamo sul set di dati _resampled!
model.fit(X_train_resampled, y_train_resampled)

import joblib
print("Salvataggio del modello su disco...")
joblib.dump(model, '/content/project/random_forest_undersampled.joblib')
print("Modello salvato.")

end_time = time.time()
print(f"Addestramento completato in {end_time - start_time:.2f} secondi.")

# --- Valutazione ---
# IMPORTANTE: valutiamo sul test set ORIGINALE!
print("\nValutazione del modello sul test set originale (sbilanciato)...")
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)

print(f"\nAccuracy sul test set: {accuracy:.4f}")
print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

# 3. Aggiungiamo un report di valutazione più dettagliato
print("\n--- Report di Classificazione Dettagliato ---")
# Questo report ci mostra le performance per ogni singola classe!
print(classification_report(y_test, predictions))
print("------------------------------------------")