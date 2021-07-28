


from numpy.core.records import array
from pandas.core.frame import DataFrame
from typing import List

from models.utils import timeWrapper

import os
from ebdeepfm.MIND_utils import utils_MIND
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


from deepctr.feature_column import SparseFeat, DenseFeat
from deepctr.models import NFM, DeepFM

import numpy as np

from models.utils import LabelEncode_sparse_features, ShareInput_wide_deep

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

from tqdm import tqdm

import nltk
from transformers import BertTokenizer
import re, string, unicodedata

from ebdeepfm.TFIDF_utils import utils_TFIDF

##########################################################################################
# Downloading and loading MIND: 
data_path = "/zhome/63/4/108196/recommenders/datasets/MIND"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "valid")
model_type = "demo"

########################
m = utils_MIND()
m.download_MIND_data(MIND_type=model_type, data_path=data_path)

# Trainset: 
behaviors = m.load_behaviors(train_path)
news = m.load_news(train_path)

# Testset: 
behaviors_test = m.load_behaviors(val_path)
news_test = m.load_news(val_path)

##########################################################################################

num_articles = 10
num_samples = 100

# Init model: 
model_tfidf = utils_TFIDF()

# Generate corpus based on titles: 
train_corpus = model_tfidf.generate_corpus_from_MIND_news_file(news_dataframe=news, num_articles=num_articles)

train_tdidf = model_tfidf.init_TFIDF(train_corpus)



# Generate embeddings based on news click historic:
behaviors_with_emb, _ = model_tfidf.transform_click_historic_to_embeddings(behaviors, news, train_tdidf, num_samples=num_samples, title=True, abstract=False, title_abstract=False)
behaviors_with_emb = pd.DataFrame(behaviors_with_emb).transpose()


# behavior_testing_format = model_tfidf.format_impressions(behaviors=behaviors, news=news, vectorizer=train_tdidf, title=True, abstract=False, title_abstract=False)


# Format the 
behavior_format = model_tfidf.format_impressions(behaviors=behaviors_with_emb, news=news, vectorizer=train_tdidf, title=True, abstract=False, title_abstract=False)
behavior_format = pd.DataFrame(behavior_format).transpose()








# ========================================================================

data = behavior_format

# Sparse features: the user_id and the article_id (equivalent to the the user-item matrix)
# Dense features: the content features of user's click history and candidate article (text data concatenated and transformed used TdIDF instance):
sparse_features = ["user_id", "article_id"]
dense_features_name = ["click_his_title_emb"] # , "target_title_abstract_emb"]

target = ["target"]

use_DeepFM = True
use_NFM = False

# behavior_format[sparse_features]
# behavior_format[target]

# ========================================================================
for dense_feature in dense_features_name:
    elements = []

    for element in data[dense_feature]: 
        elements.append(element)
    
    temp_df = pd.DataFrame(elements)
    temp_df.columns = [f"dense_feature_{temp_df.columns[i]}" for i in temp_df.columns]

    data = pd.concat([data, temp_df], axis=1)

dense_features = list(temp_df.columns)


# Sparse features:
if sparse_features is not None:  
    data[sparse_features] = LabelEncode_sparse_features(data[sparse_features], sparse_features) # Label Encode [0, ..., n]
    sparse_embeddings = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10) for feat in sparse_features] # most if models needs fixed embeddings size 
else: 
    sparse_embeddings = None

# Dense features:
if dense_features is not None:
    # data[dense_features]  = StandardScore_dense_features(data[dense_features], dense_features) # Stardize (x-mu)/sigma
    dense_embeddings  = [DenseFeat(feat, 1,) for feat in dense_features]
else: 
    dense_embeddings = None

wide_input, deep_input, feature_names = ShareInput_wide_deep(sparse_embeddings, dense_embeddings)


# 3. Generate input data for model
train, test = train_test_split(data, test_size=0.2, random_state=2021)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

print(f"Number of samples in data: {behaviors.shape[0]}")
print(f"Number of samples in training data: {train.shape[0]}")
print(f"Number of samples in test data: {test.shape[0]}")

# ========================================================================
# 4.Define Model, train, predict and evaluate
if use_DeepFM:
    model = DeepFM(wide_input, deep_input, task="binary")
if use_NFM:
    model = NFM(wide_input, deep_input, task='binary')

model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

# Fit model:
history = model.fit(train_model_input, train[target].values.astype(np.float32), batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

# Testing: 
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values.astype(np.float32), pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values.astype(np.float32), pred_ans), 4))



start_func = timeWrapper.TimeStart()
end_func = timeWrapper.TimeEnd(start_func)
timeWrapper.print_TimeTaken(end_func)
