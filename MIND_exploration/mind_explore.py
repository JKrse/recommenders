# ========================================================================

# from numpy.core.fromnumeric import shape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd 
import numpy as np
import os

from MIND_exploration.download_mind import mind_url, download_wrapper
from MIND_exploration.load_mind import load_mind
from MIND_exploration.utils import add_embeddings_to_behavior, format_impressions

from models.utils import timeWrapper
from models.utils import LabelEncode_sparse_features, ShareInput_wide_deep

from deepctr.feature_column import SparseFeat, DenseFeat
from deepctr.models import NFM, DeepFM

# ========================================================================
# ========================================================================

# Download data
_, url = mind_url("small")
data_path = "/zhome/63/4/108196/recommenders/datasets1"

temp_dir = f"{data_path}/MIND"

if not os.path.exists(temp_dir):
    download_wrapper(url, temp_dir)

# Open data: 
behaviors = load_mind.load_behaviors(temp_dir)
news = load_mind.load_news(temp_dir)

news.index = news["id"] # easy to index in the dataframe row name = newsID (all unique)

# ========================================================================

# ================================
# With too many paper, an idea is to take top X articles (e.g. last 10.000 articles) an create the vocubalary:
# ================
# Create a Corpus
    # Learn vocabulary and idf from training corpus
    # We will base the corpus on "all" the text from news file, e.g.: 
    #   corpus = news["title"]
    #   vectorizer = TfidfVectorizer()
    #   vectorizer.fit(corpus)

num_samples = 3000
num_vocab_articles = 200
train_test_frac = 0.8
num_test_samples = 1000
history_len = 100

num_train_samples = int(behaviors.shape[0] * train_test_frac)

behaviors = behaviors[:num_samples]

# OBS the TF-IDF on news.tsv file, thus, not as part either training or test set:
# (this might be a necessity to ensure that dimension does not explode)
corpus = list(news["title"][:num_vocab_articles])
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
print(f"Dimension of TF-IDF matrix: {vectorizer.transform(corpus).shape} [documents, words]")

# ================================
# ================
# Transform user's click history to document-term matrix:
    #   click_history = behaviors["history"][0]
    #   List_Of_Content_Features = Extract_Text_In_History(click_history) [e.g. extrating title]
    #   Merge_Content_Features_To_One_String = functools_reduce_iconcat(List_Of_Content_Features)
    #   user_features = vectorizer.transform(Merge_Content_Features_To_One_String)
    #   impression.append(user_features)

behaviors_with_emb, _ = add_embeddings_to_behavior(behaviors, news, vectorizer, num_samples, history_len)
behaviors_with_emb = pd.DataFrame(behaviors_with_emb).transpose()

# ================================
# ================
# Transform "Impressions" to "target" label
    # Each impression is treated as an instance (each article in an impression is a row in new format)
    # Restrict number of impressions 

behavior_format = format_impressions(behaviors_with_emb, news, vectorizer, k_samples = 7)
behavior_format = pd.DataFrame(behavior_format).transpose()

# ========================================================================
# ================ ######### ================ #
# ================ # MODEL # ================ #

data = behavior_format

# Sparse features: the user_id and the article_id (equivalent to the the user-item matrix)
# Dense features: the content features of user's click history and candidate article (text data concatenated and transformed used TdIDF instance):
sparse_features = ["user_id", "article_id"]
dense_features_name = ["abstract_title_emb"] # , "target_title_abstract_emb"]

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
print(f"Number of samples in training data: {test.shape[0]}")

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
