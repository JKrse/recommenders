# ========================================================================
# Load packages: 
from copy import Error
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

from models.utils import __missing_values, wide_deep_inputs, get_model, LabelEncode_sparse_features, StandardScore_dense_features


# ========================================================================
# Functions: 
def __binary_target_movieLens(target):
    """
    Preprocess the ratings into negative and positive samples
    Args:
        target ([int]): ratings
    Returns:
        [int]: binary ratings (0 or 1)
    """
    target[target <= 3] = 0  # ratings less than or equal to 3 classified as 0
    target[target > 3] = 1  # ratings bigger than 3 classified as 1
    return target


def wide_deep_inputs(df_data, sparse_feature_col, dense_feature_col, model_type=None): 

    sparse_embedding = [SparseFeat(feat, vocabulary_size=df_data[feat].max() + 1, embedding_dim=4) for feat in sparse_feature_col]
    dense_embedding  = [DenseFeat(feat, 1,) for feat in dense_feature_col]

    feature_names = get_feature_names(sparse_embedding + dense_embedding)

    if model_type is not None:
        if model_type is "DeepFM" or "NFM":
            wide_input = sparse_embedding + dense_embedding
            deep_input = wide_input
            feature_names = get_feature_names(sparse_embedding + dense_embedding)
        else: 
            raise NameError("Model type is not defined")

    return wide_input, deep_input, feature_names


def get_model(model_type="DeepFM", wide_input=None, deep_input=None, task="classification"):

    if wide_input is None or deep_input is None:
        raise NameError("Missing Wide or Deep or input")

    if model_type is "DeepFM":
        model = DeepFM(wide_input, deep_input, task)
    elif model_type is "NFM":
        model = NFM(wide_input, deep_input, task)
    else:
        raise NameError("Model type is not defined")    

    if task is "classification": 
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
    elif task is "regression": 
        model.compile("adam", "mse", metrics=['mse'], )
    else:
        raise NameError("Task is not defined")

    return model


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
data = LabelEncode_sparse_features(data, sparse_features)
data = StandardScore_dense_features(data, dense_features)

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

history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))


