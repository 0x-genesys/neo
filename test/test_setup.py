#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("Python ML Project Setup Test")
print("=" * 30)

# Test numpy
print("1. NumPy test:")
arr = np.array([1, 2, 3, 4, 5])
print(f"   Created array: {arr}")
print(f"   Mean: {np.mean(arr)}")

# Test pandas
print("\n2. Pandas test:")
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(f"   Created DataFrame:\n{df}")

# Test scikit-learn
print("\n3. Scikit-learn test:")
iris = datasets.load_iris()
print(f"   Iris dataset loaded: {iris.data.shape[0]} samples, {iris.data.shape[1]} features")

# Test matplotlib
print("\n4. Matplotlib test:")
print(f"   Matplotlib version: {matplotlib.__version__}")

# Test seaborn
print("\n5. Seaborn test:")
print(f"   Seaborn version: {sns.__version__}")

# Create a simple plot
plt.figure(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Plot: Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.savefig("test_plot.png")
plt.close()
print("   Test plot saved as test_plot.png")

print("\n✅ All libraries imported successfully!")
print("\nNote: TensorFlow is not installed because it doesn't support Python 3.14 yet.")
print("Consider using PyTorch or waiting for TensorFlow compatibility updates.")