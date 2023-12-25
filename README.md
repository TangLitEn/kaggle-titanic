# kaggle-titanic
## Titanic Dataset Analysis and Model Exploration

### Introduction
This notebook is a comprehensive exploration of the Titanic dataset. The primary focus is on data analysis, visualization, and applying various machine learning models to predict survival rates.

### Libraries Used
The following Python libraries are used in this notebook:
- NumPy
- pandas
- Matplotlib
- seaborn
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Strategy
The approach taken in this analysis involves several steps:
1. Splitting the training data into two datasets: `real_train` and `test_train`.
2. Training different models using `real_train` and testing the models on `test_train`.
3. Retraining the best model on the full `real_train` dataset.
4. Using the final model to test on `test_final`.

### Data Reading
Data is read from a CSV file into a pandas DataFrame.
```python
train_data = pd.read_csv("data/train.csv")
train_data.head()
```

### Data Analysis
Key factors explored in the analysis include:
1. Impact of Gender on Survival Rate.
2. Influence of Passenger Class (PClass) on Survival Rate.
3. Effect of Age on Survival Rate.
4. Relationship between Siblings/Spouses aboard (SibSp) and Survival Rate.
5. Connection between Parents/Children aboard (Parch) and Survival Rate.

### Model Exploration
The following models are explored:
- K-Nearest Neighbours
- Decision Trees
- Support Vector Machines
- Neural Networks

The process for each model includes:
1. Duplicating dataset.
2. Data pre-processing.
3. Splitting dataset into training and testing sets.
4. Training models using the training dataset.
5. Applying the models on the testing data.
6. Evaluating the performance of each model.

### Special Considerations
- **K-Nearest Neighbours (KNN):** Prior to applying KNN, the data is compressed using Principal Component Analysis (PCA) as KNN can only handle 2D data.

### Code Snippets
- Principal Component Analysis (PCA) is applied as follows:
```python
PCA = real_train.copy()
```
- Conversion functions for categorical data:
```python
def Sex_to_Num(row):
    if (row.Sex == "male"): return 1
    elif (row.Sex == "female"): return 0
    else: return row.Sex

PCA["Sex"] = PCA.apply(Sex_to_Numaxis=1)

def Cabin_to_Number(row):
    if (not pd.isna(row.Cabin)):
        return ord(row.Cabin[0])
    else: return 0

PCA["Cabin"] = PCA.apply(Cabin_to_Numberaxis=1)
```
- Machine Learning libraries used for PCA:
```python
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
```

### Conclusion
This notebook provides an in-depth analysis of the Titanic dataset using various statistical and machine learning techniques. The models are evaluated to determine their effectiveness in predicting survival rates based on the available data.
