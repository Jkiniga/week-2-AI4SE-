# week-2-AI4SE-
AI DRIVEN SOLUTION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data (for demonstration)
np.random.seed(42)
n_samples = 200
rainfall = np.random.uniform(50, 300, n_samples)    # mm
temperature = np.random.uniform(15, 35, n_samples)  # Celsius
soil_quality = np.random.uniform(1, 10, n_samples)  # index
# Simulated yield (with some noise)
yield_ = 2 * rainfall + 3 * temperature + 5 * soil_quality + np.random.normal(0, 50, n_samples)

data = pd.DataFrame({
    'rainfall': rainfall,
    'temperature': temperature,
    'soil_quality': soil_quality,
    'yield': yield_
})

# Save to CSV
data.to_csv('sample_farm_data.csv', index=False)

# Step 2: Split data
X = data[['rainfall', 'temperature', 'soil_quality']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train neural network
model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 5: Plot results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data (for demonstration)
np.random.seed(42)
n_samples = 200
rainfall = np.random.uniform(50, 300, n_samples)    # mm
temperature = np.random.uniform(15, 35, n_samples)  # Celsius
soil_quality = np.random.uniform(1, 10, n_samples)  # index
# Simulated yield (with some noise)
yield_ = 2 * rainfall + 3 * temperature + 5 * soil_quality + np.random.normal(0, 50, n_samples)

data = pd.DataFrame({
    'rainfall': rainfall,
    'temperature': temperature,
    'soil_quality': soil_quality,
    'yield': yield_
})

# Save to CSV
data.to_csv('sample_farm_data.csv', index=False)

# Step 2: Split data
X = data[['rainfall', 'temperature', 'soil_quality']]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train neural network
model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 5: Plot results
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs Predicted Crop Yield')
plt.show()
