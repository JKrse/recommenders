# ========================================================================

from deepctr.models import NFM, DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat
from sklearn.model_selection import train_test_split

from models.utils import LabelEncode_sparse_features, StandardScore_dense_features, ShareInput_wide_deep, __missing_values, timeWrapper
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score

from models.MovieLensDataset import MovieLensDataset

# ========================================================================
use_DeepFM = True
use_NFM = False

MovieLens = False
Criteo = True

if MovieLens:
    # Load Dataset: 
    movielens = MovieLensDataset("datasets/ml-latest-small")
    movielens.items["genres"] = [movielens.items["genres"][i].split("|")[0] for i in movielens.items.index]

    # Define sparse, dense, & target column:
    sparse_features = ["userId", "movieId", "genres"] # dummy example, Lars and I just added the first genre from the genre list
    dense_features = None
    target = "rating"

    # MovieLens setup: 
    items = movielens.items
    target_movie = movielens.target_binary
    data = pd.concat([items, target_movie], axis=1)

if Criteo:
    # Load Dataset: 
    data = pd.read_csv("examples/criteo_sample.txt")
    
    # Define sparse, dense, & target column:
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = "label"


# Simple missing value: 
data = __missing_values(data, sparse_features, dense_features)


# ========================================================================
# ========================================================================
# 2. Prepare data:

# Sparse features:
if sparse_features is not None:  
    data[sparse_features] = LabelEncode_sparse_features(data[sparse_features], sparse_features) # Label Encode [0, ..., n]
    sparse_embeddings = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10) for feat in sparse_features] # most if models needs fixed embeddings size 
else: 
    sparse_embeddings = None

# Dense features:
if dense_features is not None:
    data[dense_features]  = StandardScore_dense_features(data[dense_features], dense_features) # Stardize (x-mu)/sigma
    dense_embeddings  = [DenseFeat(feat, 1,) for feat in dense_features]
else: 
    dense_embeddings = None

wide_input, deep_input, feature_names = ShareInput_wide_deep(sparse_embeddings, dense_embeddings)

# ========================================================================
# 3. Generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2021)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

# ========================================================================
# 4.Define Model, train, predict and evaluate
if use_DeepFM:
    model = DeepFM(wide_input, deep_input, task="binary")
if use_NFM:
    model = NFM(wide_input, deep_input, task='binary')

model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

start = timeWrapper.TimeStart()
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
end = timeWrapper.TimeEnd(start)

# Print time:
timeWrapper.print_TimeTaken(end)

# Testing: 
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
