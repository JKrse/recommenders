# ========================================================================

import pandas as pd 
import os

from MIND_exploration.download_mind import mind_url, download_wrapper
from MIND_exploration.load_mind import load_mind
from models.utils import timeWrapper

from sklearn.feature_extraction.text import TfidfVectorizer

# ========================================================================

import functools
import operator
def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, )

# ========================================================================

# Download data
url = mind_url("small_dev")
data_path = "/zhome/63/4/108196/recommenders/datasets"

temp_dir = f"{data_path}/MIND"

if not os.path.exists(temp_dir):
    download_wrapper(url, temp_dir)

# Open data: 
behaviors = load_mind.load_behaviors(temp_dir)
news = load_mind.load_news(temp_dir)

# ========================================================================


# Create a Corpus
    # Learn vocabulary and idf from training corpus
    # We will base the corpus on "all" the text from news file, e.g.: 
    
    #   corpus = news["title"]
    #   vectorizer = TfidfVectorizer()
    #   vectorizer.fit(corpus)
    
# Transform user's click history to document-term matrix: 

    #   click_history = behaviors["history"][0]
    #   List_Of_Content_Features = Extract_Text_In_History(click_history) [e.g. extrating title]
    #   Merge_Content_Features_To_One_String = functools_reduce_iconcat(List_Of_Content_Features)
    #   user_features = vectorizer.transform(Merge_Content_Features_To_One_String)


# MANGLER -- TARGET
    # Lav target: impression --> eventuel lav negativ sampling. 
    # 

# MANGLER -- EVALUATION

num_samples = 200
corpus = list(news["title"][0:num_samples])

vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)
vectorizer.transform(corpus)

X = vectorizer.fit_transform(corpus)
X.shape



test_corpus = list(news["title"][num_samples:num_samples+100])


test_corpus_dummy = functools_reduce_iconcat(test_corpus)




test = vectorizer.transform([test_corpus_dummy])
test.shape


start_func = timeWrapper.TimeStart()
end_func = timeWrapper.TimeEnd(start_func)
timeWrapper.print_TimeTaken(end_func)
