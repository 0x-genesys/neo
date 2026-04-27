#!/usr/bin/env python3
"""
Example ML script using the installed libraries.
Demonstrates basic data manipulation, visualization, and machine learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Machine Learning Example")
print("=" * 50)

# 1. Create synthetic dataset
print("\n1. Creating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

print(f"   Dataset shape: {X.shape}")
print(f"   Target classes: {np.unique(y)}")

# 2. Convert to pandas DataFrame for exploration
print("\n2. Data exploration with pandas...")
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"   DataFrame shape: {df.shape}")
print(f"   First 5 rows:\n{df.head()}")
print(f"   Basic statistics:\n{df.describe().round(2)}")

# 3. Split data
print("\n3. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# 4. Train model
print("\n4. Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("   Model trained successfully!")

# 5. Evaluate model
print("\n5. Model evaluation...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Classification Report:\n{classification_report(y_test, y_pred)}")

# 6. Feature importance
print("\n6. Feature importance analysis...")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("   Top 5 most important features:")
for i in range(5):
    print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 7. Visualization
print("\n7. Creating visualizations...")

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("   Saved: feature_importance.png")

# Correlation heatmap (first 10 features)
plt.figure(figsize=(10, 8))
corr_matrix = df.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap (First 10 Features)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("   Saved: correlation_heatmap.png")

plt.close('all')
print("\n✅ Example completed successfully!")
print("\nGenerated files:")
print("  - feature_importance.png")
print("  - correlation_heatmap.png")
print("\nTo run this example:")
print("  source venv/bin/activate")
print("  python example.py")