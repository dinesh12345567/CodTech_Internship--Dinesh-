# ===== Task 1: Data Pipeline Development =====

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# === Step 1: Load Dataset ===
data_path = "Personal_Finance_and_Spendings.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("First few rows:\n", df.head(), "\n")

# === Step 2: Feature / Target Split ===
target_column = "Rent"
X = df.drop(columns=[target_column])
y = df[target_column]

# === Step 3: Detect Data Types ===
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features, "\n")

# === Step 4: Build Pipelines ===
# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# === Step 5: Combine Pipelines ===
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# === Step 6: Apply Transformation ===
X_transformed = preprocessor.fit_transform(X)
print(f"Data shape after preprocessing: {X_transformed.shape}\n")

# === Step 7: (Optional) Dimensionality Reduction ===
# You can skip this step if PCA is not needed
pca = PCA(n_components=min(X_transformed.shape[1], 5))
X_reduced = pca.fit_transform(X_transformed)
print(f"Shape after PCA reduction: {X_reduced.shape}\n")

# === Step 8: Merge Processed Data with Target ===
processed_features = pd.DataFrame(X_reduced)
processed_target = pd.DataFrame(y, columns=[target_column])
final_data = pd.concat([processed_features, processed_target], axis=1)

# === Step 9: Save Processed Data ===
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "processed_data.csv")
final_data.to_csv(output_file, index=False)

print(f"✅ Processed dataset saved at: {output_file}")

