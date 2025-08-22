import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
import time
from imblearn.under_sampling import RandomUnderSampler
import joblib

df = pd.read_csv('/content/project/training_dataset_completo.tsv', sep='\t')
df.dropna(subset=['Interaction'], inplace=True)

numerical_features = ['s_rsa', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_rsa', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']
categorical_features = ['s_resn', 't_resn', 's_ss8', 't_ss8']
target_column = 'Interaction'

y = df[target_column]
X_features = df[numerical_features + categorical_features].copy()

for col in numerical_features:
    mean_value = X_features[col].mean()
    X_features[col] = X_features[col].fillna(mean_value)
X = pd.get_dummies(X_features, columns=categorical_features, dummy_na=False)

model_columns = X.columns
joblib.dump(model_columns, '/content/project/model_columns.joblib')

print(f"Features final number: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts())

# Model training
model = RandomForestClassifier(
    n_estimators=100,      
    random_state=42,       
    n_jobs=-1
)

start_time = time.time()
model.fit(X_train_resampled, y_train_resampled)

import joblib
joblib.dump(model, '/content/project/random_forest_undersampled.joblib')

end_time = time.time()

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)

print(f"\nAccuracy on test set: {accuracy:.4f}")
print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

print(classification_report(y_test, predictions))
