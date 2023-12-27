# Titanic Survival Analysis

## Introduction
This project focuses on analyzing the Titanic dataset to identify factors influencing passenger survival. Utilizing various data science techniques, we aim to build a predictive model to estimate survival outcomes based on passenger characteristics.

## Strategy
Our approach involves splitting the dataset into two: `real_train` for model training and `test_train` for validation. This ensures robust model evaluation and tuning.

### Data Reading
The data is loaded using pandas:
```python
import pandas as pd
data = pd.read_csv('titanic_data.csv')
```

## Data Analysis
We conducted an exploratory analysis to understand the data and uncover patterns related to survival.

### Investigating Factors
#### Gender Impact
Analysis revealed a significant correlation between gender and survival:
```python
data.groupby('Sex')['Survival'].mean().plot(kind='bar')
```

#### Class Impact
Passenger class also showed a noticeable effect on survival rates:
```python
data.groupby('Pclass')['Survival'].mean().plot(kind='bar')
```

#### Age Impact
Age's role was explored, highlighting varying survival rates across age groups:
```python
data.groupby(pd.cut(data['Age'], bins=5))['Survival'].mean().plot(kind='bar')
```

## Model Exploration
We experimented with multiple models to find the best predictor of survival.

### K-Nearest Neighbours (KNN)
We compressed data using PCA before applying KNN:
```python
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(transformed_data, data['Survival'])
```

### Decision Trees
#### Single Tree Analysis
We analyzed the effectiveness of a single decision tree.
```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(train_data, train_labels)
```

#### Random Forest
Random Forests were used with parameter tuning for better accuracy.
```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, max_depth=5)
forest.fit(train_data, train_labels)
```

### Neural Network
Implemented a simple neural network using TensorFlow:
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10)
```

## Training the Final Model
We selected the Random Forest model based on its performance and generalizability:
```python
final_model = RandomForestClassifier(n_estimators=100, max_depth=5)
final_model.fit(real_train_data, real_train_labels)
```

## Generating Test Results
Predictions on the test dataset were made using the final model:
```python
predictions = final_model.predict(test_train_data)
```

## Conclusion
The analysis highlighted key factors like gender, class, and age in survival prediction. The Random Forest model provided the best balance of accuracy and generalizability.

## Future Work
Future enhancements could include integrating more features, trying advanced models, and using larger datasets for more robust predictions.

## How to Run the Notebook
1. Ensure Python and Jupyter Notebook are installed.
2. Install necessary packages: `pandas`, `sklearn`, `matplotlib`, `tensorflow`.
3. Run the notebook cell by cell to observe each step of the analysis and modeling.
