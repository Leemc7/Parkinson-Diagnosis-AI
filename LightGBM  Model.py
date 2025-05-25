# ---------------------------------------------------
# lightgbm_parkinsons.py
# ---------------------------------------------------

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the data
# Make sure "parkinsons.data" has a header row matching the columns below.
# If the file doesn't have headers, you'll need to supply the columns names explicitly:
# For reference, the column names (as typically provided) are:
#
#   name, MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs),
#   MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3,
#   Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, status, RPDE, DFA, spread1,
#   spread2, D2, PPE
#
# If your downloaded file does not have this header row, uncomment and pass these to `names=...`
# columns = [
#     "name","MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)",
#     "MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
#     "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1",
#     "spread2","D2","PPE"
# ]
# data = pd.read_csv("parkinsons.data", names=columns)

data = pd.read_csv("parkinsons.data")

# 2. Inspect and clean the data
# Drop the 'name' column, since it's not a useful feature
if 'name' in data.columns:
    data.drop('name', axis=1, inplace=True)

# The target column is 'status' (1 = Parkinson’s, 0 = healthy)
X = data.drop('status', axis=1)
y = data['status']

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Keep class proportions the same
)

# 4. Create and train the LightGBM model
# You can tune parameters such as num_leaves, n_estimators, learning_rate, etc.
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

# Before fitting the model, replace invalid characters in column names:
X_train.columns = X_train.columns.str.replace(r'[,;:\[\]\{\}\"\']', '', regex=True) 
X_test.columns = X_test.columns.str.replace(r'[,;:\[\]\{\}\"\']', '', regex=True) 

model.fit(X_train, y_train)

# 5. Predictions and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
