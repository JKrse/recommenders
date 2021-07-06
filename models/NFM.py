# ========================================================================

from deepctr.models import NFM, DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split

from models.utils import transform_sparse_features, transform_dense_features, wide_deep_inputs
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

from models.MovieLensDataset import MovieLensDataset

# ========================================================================

def ShareInput_wide_deep(sparse_embedding, dense_embedding):
        
    if sparse_embedding is not None and dense_embedding is not None: 
        wide_input = sparse_embedding + dense_embedding
        deep_input = wide_input
    elif sparse_embedding is not None and dense_embedding is None:
        wide_input = sparse_embedding
        deep_input = wide_input
    elif sparse_embedding is None and dense_embedding is not None:
        wide_input = dense_embedding
        deep_input = wide_input
    
    feature_names = get_feature_names(wide_input + deep_input)
    
    return wide_input, deep_input, feature_names

# ========================================================================

movielens = MovieLensDataset("datasets/ml-latest-small")
movielens.items["genres"] = [movielens.items["genres"][i].split("|")[0] for i in movielens.items.index]

sparse_features = ["userId", "movieId", "genres"]
dense_features = None
target_name = "rating"

items = movielens.items
target = movielens.target_binary
data = pd.concat([items, target], axis=1)

# ========================================================================
# === Class : 

# Sparse features:
if sparse_features is not None: 
    data[sparse_features] = transform_sparse_features(data[sparse_features], sparse_features) # Label Encode [0, ..., n]
    sparse_embedding = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10) for feat in sparse_features]
else: 
    sparse_embedding = None

# Dense features:
if dense_features is not None:
    data[dense_features]  = transform_dense_features(data, dense_features) # Stardize (x-mu)/sigma
    dense_embedding  = [DenseFeat(feat, 1,) for feat in dense_features]
else: 
    dense_embedding = None

wide_input, deep_input, feature_names = ShareInput_wide_deep(sparse_embedding, dense_embedding)



# 3. Generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2021)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

# ========================================================================
# 4.Define Model, train, predict and evaluate

model = DeepFM(wide_input, deep_input, task="binary")
model = NFM(wide_input, deep_input, task='binary')

model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target_name].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target_name].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target_name].values, pred_ans), 4))








