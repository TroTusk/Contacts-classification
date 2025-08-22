import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, matthews_corrcoef

df = pd.read_csv('training_dataset_completo.tsv', sep='\t')

# Remove rows with empty column (NaN).
df.dropna(subset=['Interaction'], inplace=True)

# Features
numerical_features = ['s_rsa', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_rsa', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']
categorical_features = ['s_resn', 't_resn', 's_ss8', 't_ss8']

target_column = 'Interaction'

y = df[target_column]

X_features = df[numerical_features + categorical_features]

X_features = X_features.copy()

# Manage NaN
for col in numerical_features:
    X_features[col].fillna(X_features[col].mean(), inplace=True)
    
X_processed = pd.get_dummies(X_features, columns=categorical_features, dummy_na=False)

X = X_processed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)
mcc = matthews_corrcoef(y_test, predictions)

print(f"\nAccuracy on test set: {accuracy:.4f}")
print(f"Matthew's Correlation Coefficient: {mcc:.4f}")

# Example of a prediction
print("\n--- Example ---")
example_features = X_test.iloc[0] 
predicted_class = predictions[0]
actual_class = y_test.iloc[0]

print("{predicted_class}")
print("{actual_class}")

class_order = model.classes_
probabilities_for_example = probabilities[0]

print("\n--- Probability ---")
for i in range(len(class_order)):
    print(f"- {class_order[i]}: {probabilities_for_example[i]:.4f}")
