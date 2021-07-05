# ========================================================================
# Load packages: 
from copy import Error
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

from functions import __binary_target_movieLens, __missing_values, wide_deep_inputs, get_model

# ========================================================================
# Functions: 

# ========================================================================

# Get data: 
#file_path = "examples/criteo_sample.txt"
file_path = "examples/movielens_sample.txt"

task = "classification"
model_type = "DeepFM"

# Define Sparse and Dense features: 
sparse_features = ["user_id", "movie_id"]
dense_features  = ["age", "occupation"]

# Define Target value:
target = ["rating"]

data = pd.read_csv(file_path)

# MovieLens:
data[target] = __binary_target_movieLens(data[target])

# If missing values: 
data = __missing_values(data, sparse_features, dense_features)


# ========================================================================
# 1. Label Encoding for sparse features, and standardize Transformation for dense features 
for feat in sparse_features:
    lbe = LabelEncoder() 
    data[feat] = lbe.fit_transform(data[feat])

scaler = StandardScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])


# ========================================================================
# 2. Count #unique features for each sparse field, and record dense feature field name
wide_input, deep_input, feature_names = wide_deep_inputs(data, sparse_features, dense_features, model_type="DeepFM")


# ========================================================================
# 3. Generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2021)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}


# ========================================================================
# 4.Define Model, train, predict and evaluate

model = get_model("DeepFM", wide_input, deep_input, task="classification")

history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=100, verbose=2, validation_split=0.2, )

pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

