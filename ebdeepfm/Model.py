

from numpy.core.records import array
from pandas.core.frame import DataFrame

from models.utils import timeWrapper

import os
from ebdeepfm.MIND_utils import utils_MIND
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


import functools
import operator


from deepctr.feature_column import SparseFeat, DenseFeat
from deepctr.models import NFM, DeepFM

import numpy as np

from models.utils import LabelEncode_sparse_features, ShareInput_wide_deep

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

from tqdm import tqdm

def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, )


class utils_TFIDF: 
    """Super for deepCTR models
    """


    def __init__(self) -> None:
        pass


    def init_TFIDF(self, corpus: list = None) -> None:
        """[summary]
        Args:
            corpus (list, str): List of strings. Defaults to None.
        """
        
        vectorizer = TfidfVectorizer()
        tdidf = vectorizer.fit(corpus)
        # tdidf = vectorizer.fit_transform(corpus)
        print(f"Dimension of TF-IDF matrix: {vectorizer.transform(corpus).shape} [documents, words]")

        return tdidf
    

    def transform_input_to_document_term_matrix_array(self, raw_document: list, TFIDF_Vectorizer=None) -> None: # numpy array
        """[summary]

        Args:
            raw_document (list, str): list of strings (documents)
            TFIDF_Vectorizer ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        return TFIDF_Vectorizer.transform(raw_document).toarray()
    

    def transform_click_historic_to_embeddings(self, behaviors: DataFrame, news: DataFrame, vectorizer, num_samples=None, max_history_len=50, title:bool=True, abstract:bool=False, title_abstract:bool=False):
        """Make a dictionary that contain all features of behaviors with the content features. These can come from either title, abstract, and title-abstract.
        The content features are based on the TF-IDF (vectorizer) that have been fitted to fit().

        Args:
            behaviors (([DataFrame]): uses the columns "history", "title", "abstract" in behaviors.tsv
            news ([DataFrame]): the news.tsv file.
            vectorizer ([TfidfVectorizer]): sklearn.feature_extraction.text.TfidfVectorizer object defined in .tokenize_text() (that has been fitted to a corpus).
            num_samples (int): number of samples to be used in the Dataframe. Defaults to None (i.e. all).
            max_history_len (int, optional): user's click history. Defaults to 50.
            title (bool, optional): Generate concatenated click history embeddings based on title. Defaults to True.
            abstract (bool, optional): Generate concatenated click history embeddings based on abstract. Defaults to False.
            title_abstract (bool, optional): Generate concatenated click history embeddings based on title and abstract. Defaults to False (if true both title and abstract are set to True).

        Returns:
            data_dict [dict]]:  Dictionary similar to behaviors file but with TF-IDF features for a given click historic (These are based on either title, abstract or title-abstract)
                        To get dataframe: pd.DataFrame(data_dict).transpose()
        """
        
        news.index = news["id"]
        no_history = 0
        data_dict = {}

        if num_samples is None: 
            num_samples = len(behaviors)
        
        if title_abstract:
            title=True
            abstract=True

        for impression in tqdm(range(num_samples)):
            
            history = behaviors.loc[impression]["history"]

            try: 
                # If user has a click history:
                history = history.split(" ")[0:max_history_len]

                titles = []
                abstracts = []

                for article in history: 
                    if title:
                        titles.append(str(news.loc[article, :]["title"]))
                    if abstract:
                        abstracts.append(str(news.loc[article, :]["abstract"]))
                
                # TODO 
                # data clean should be done...

                if title:
                    title_text = functools_reduce_iconcat(titles)
                if abstract:
                    abstract_text = functools_reduce_iconcat(abstracts)
                if title_abstract:
                    title_abstract_text = title_text + " " + abstract_text
            except:
                # If the user does not have a click history:
                title_text=""
                abstract_text=""
                no_history += 1 
                # print(f"ImpressionID {impression} has no click history.")

            data_dict[impression] = {key : behaviors.loc[impression][key] for key in behaviors}

            if title:
                data_dict[impression]["click_his_title_emb"] = vectorizer.transform([title_text]).toarray()[0]
            if abstract:
                data_dict[impression]["click_his_abstract_emb"] = vectorizer.transform([abstract_text]).toarray()[0]
            if title_abstract:
                data_dict[impression]["click_his_abstract_title_emb"] = vectorizer.transform([title_abstract_text]).toarray()[0]

        return data_dict, no_history
        


    def format_impressions(self, behaviors:DataFrame, news:DataFrame, vectorizer, k_samples: int= 7, title:bool=True, abstract:bool=False, title_abstract:bool=False) -> dict:
        """Format the behaviors files so that each impression is a row with a target value (1 or 0) 

        Args:
            behaviors ([DataFrame]): behaviors.tsv file. Will format data in "impressions" column.
            news ([DataFrame]): the news.tsv file. 
            vectorizer ([TfidfVectorizer]): sklearn.feature_extraction.text.TfidfVectorizer object defined in .tokenize_text() (that has been fitted to a corpus).
            k_samples (int, optional): Number of negative samples of impression log to add, as there can be quite a few (simple "negative sampling"). Defaults to 7.
            title (bool, optional): Generate concatenated click history embeddings based on title. Defaults to True.
            abstract (bool, optional): Generate concatenated click history embeddings based on abstract. Defaults to False.
            title_abstract (bool, optional): Generate concatenated click history embeddings based on title and abstract. Defaults to False (if true both title and abstract are set to True).

        Returns:
            dict: dictionary with new format and impression article embedding
        """

        data_dict = {}
        sample_no = 0 

        if title_abstract:
            title=True
            abstract=True
        
        for index in tqdm(behaviors.index):
            
            samples = behaviors.loc[index]["impressions"].split(" ")

            for i, sample in enumerate(samples): 
                
                if "-1" in sample or i < k_samples:
                    temp = sample.split("-")
                    # Temporal
                    if title:
                        temp_title = str(news.loc[temp[0]]["title"])
                        target_title_emb = vectorizer.transform([temp_title]).toarray()[0]
                    if abstract:
                        temp_abstract = str(news.loc[temp[0]]["abstract"])
                        target_abstract_emb = vectorizer.transform([temp_abstract]).toarray()[0]
                    if title_abstract:
                        temp_title_abstract = temp_title + " " + temp_abstract
                        target_title_abstract_emb = vectorizer.transform([temp_title_abstract]).toarray()[0]
                    

                    # Copy existings column names:
                    data_dict[sample_no] = {key : behaviors.loc[index][key] for key in behaviors}
                    
                    # Add new column names:
                    data_dict[sample_no]["article_id"] = temp[0]
                    if title:
                        data_dict[sample_no]["target_title_emb"] = target_title_emb
                    if abstract:
                        data_dict[sample_no]["target_abstract_emb"] = target_abstract_emb
                    if title_abstract:
                        data_dict[sample_no]["target_title_abstract_emb"] = target_title_abstract_emb
                    
                    data_dict[sample_no]["target"] = int(temp[1])
                    
                    sample_no += 1

        return data_dict

##########################################################################################
# Downloading and loading MIND: 
data_path = "/zhome/63/4/108196/recommenders/datasets/MIND"
train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "valid")

m = utils_MIND()
m.download_MIND_data(MIND_type="demo", data_path=data_path)

behaviors = m.load_behaviors(train_path)
news = m.load_news(train_path)

news_test = m.load_news(val_path)

# Generate corpus based on titles: 
train_corpus = m.generate_corpus_from_MIND_news_file(news_dataframe=news, num_articles=300)
# test_corpus = m.generate_corpus_from_MIND_news_file(news_dataframe=news_test, num_articles=200)

##########################################################################################

model_tfidf = utils_TFIDF()
train_tdidf = model_tfidf.init_TFIDF(train_corpus)


behaviors_with_emb, _ = model_tfidf.transform_click_historic_to_embeddings(behaviors, news, train_tdidf, num_samples=10000, title=True, abstract=False, title_abstract=False)
behaviors_with_emb = pd.DataFrame(behaviors_with_emb).transpose()


start_func = timeWrapper.TimeStart()

behavior_format = model_tfidf.format_impressions(behaviors=behaviors_with_emb, news=news, vectorizer=train_tdidf, title=True, abstract=False, title_abstract=False)
behavior_format = pd.DataFrame(behavior_format).transpose()

end_func = timeWrapper.TimeEnd(start_func)
timeWrapper.print_TimeTaken(end_func)



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
