# EXNO:4-DS
# NAME : SHREEJA R S
# REF.NO : 25017561

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd

data = pd.read_csv("bmi.csv")

data.head()

```
<img width="770" height="406" alt="Screenshot 2026-03-11 153847" src="https://github.com/user-attachments/assets/e13aea06-7092-42da-b4ed-e934bd15c148" />


```
data = pd.read_csv("income.csv")

data.head()
```
<img width="1868" height="459" alt="Screenshot 2026-03-11 153901" src="https://github.com/user-attachments/assets/70c49345-bdff-4ff1-b850-6d76d082f69d" />

```
# Remove duplicate values
data = data.drop_duplicates()

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

data.head()
```
<img width="1856" height="571" alt="Screenshot 2026-03-11 153919" src="https://github.com/user-attachments/assets/edfa9b6e-2023-4654-820c-b53e2cc48896" />

```
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
```


```
numeric_data = data.select_dtypes(include=np.number)

numeric_data.head()
```

<img width="916" height="586" alt="Screenshot 2026-03-11 153938" src="https://github.com/user-attachments/assets/eb807038-c5d1-4af4-ad5c-e56f05175835" />

```
scaler1 = StandardScaler()

standard_scaled = scaler1.fit_transform(numeric_data)

standard_scaled = pd.DataFrame(
standard_scaled,
columns=numeric_data.columns)

standard_scaled.head()
```
<img width="915" height="493" alt="Screenshot 2026-03-11 153951" src="https://github.com/user-attachments/assets/518496dc-94a5-41be-8e56-f770392e77ec" />

```
scaler2 = MinMaxScaler()

minmax_scaled = scaler2.fit_transform(numeric_data)

minmax_scaled = pd.DataFrame(
minmax_scaled,
columns=numeric_data.columns)

minmax_scaled.head()
```
<img width="908" height="489" alt="Screenshot 2026-03-11 154006" src="https://github.com/user-attachments/assets/b9d3ebe7-6ea8-4103-a5ed-56be49926a60" />

```
scaler3 = MaxAbsScaler()

maxabs_scaled = scaler3.fit_transform(numeric_data)

maxabs_scaled = pd.DataFrame(
maxabs_scaled,
columns=numeric_data.columns)

maxabs_scaled.head()
```
<img width="877" height="511" alt="Screenshot 2026-03-11 154020" src="https://github.com/user-attachments/assets/e34c5ec9-55b1-4100-84fe-8742c5b03cf2" />

```
scaler4 = RobustScaler()

robust_scaled = scaler4.fit_transform(numeric_data)

robust_scaled = pd.DataFrame(
robust_scaled,
columns=numeric_data.columns)

robust_scaled.head()
```
<img width="925" height="474" alt="Screenshot 2026-03-11 154045" src="https://github.com/user-attachments/assets/9faa87be-5d00-4bc7-8b99-2fd9a6326f43" />

```
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X = numeric_data.iloc[:, :-1]

y = numeric_data.iloc[:, -1]

selector = SelectKBest(score_func=f_regression,
k=2)

X_new = selector.fit_transform(X,y)

selected_features = X.columns[selector.get_support()]

print("Selected Features:")
print(selected_features)
```

<img width="919" height="505" alt="Screenshot 2026-03-11 154103" src="https://github.com/user-attachments/assets/ce5cca0b-eeaa-4544-b249-3355d24a89a8" />

```
final_data = pd.concat([pd.DataFrame(X_new),y],axis=1)

final_data.to_csv("processed_data.csv",index=False)

print("File Saved Successfully")
```

<img width="973" height="243" alt="Screenshot 2026-03-11 154115" src="https://github.com/user-attachments/assets/0b4df305-6b4a-4ff5-8834-7c3ee2f0e8da" />

# RESULT:
Thus the given dataset is read successfully.

Data cleaning process is completed.

Feature scaling techniques such as StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler are applied.

Feature selection is performed successfully.

The processed data is saved to a file.
