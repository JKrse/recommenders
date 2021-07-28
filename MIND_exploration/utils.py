# ========================================================================

import functools
import operator

# ========================================================================

def join_lists_of_stings(a):
    """[summary]

    Args:
        a ([type]): [description]

    Returns:
        [type]: [description]
        > For example: 
        > join_lists_of_stings(["The 25 US cities where it's easiest to get a mortgage", "NYPD Commissioner James O'Neill To Resign: Reports"])
        > Output: ["The 25 US cities where it's easiest to get a mortgage NYPD Commissioner James O'Neill To Resign: Reports"]
        """
        
    return " ".join(a) 

# The following code is not written with production in mind, rather being functional

def add_embeddings_to_behavior(behaviors, news, vectorizer, num_samples=None, history_len=100):
    """    
    Make a dictionary that contain all features of behaviors with the content features from title and abstract.
    The content features are based on the TF-IDF (vectorizer)

    Args:
        behaviors ([DataFrame]): uses the columns "history", "title", "abstract" in behaviors.tsv
        news ([DataFrame]): the news.tsv file.
        vectorizer ([TfidfVectorizer]): the TF-IDF model (that has been fitted to a corpus).
        num_samples (int): number of samples to be used in the Dataframe. Defaults to None (i.e. all).
        history_len (int,): user's click history. Defaults to 100.

    Returns:
        data_dict [dict]]:    dictionary similar to behaviors but with TF-IDF features for title and abstract features for a given click historic
                    to get dataframe: pd.DataFrame(data_dict).transpose()
    """
    
    news.index = news["id"]
    no_history = 0
    data_dict = {}

    if num_samples is None: 
        num_samples = len(behaviors)
    
    for impression in range(num_samples):
        
        history = behaviors.loc[impression]["history"]

        try: 
            # If user has a click history:
            history = history.split(" ")[0:history_len]

            titles = []
            abstracts = []

            for article in history: 
                titles.append(str(news.loc[article, :]["title"]))
                abstracts.append(str(news.loc[article, :]["abstract"]))
            
            # data clean should be done...
            title_text = join_lists_of_stings(titles)
            abstract_text = join_lists_of_stings(abstracts)
            title_abstract_text = title_text + " " + abstract_text
        except:
            # If the user does not have a click history:
            title_text=""
            abstract_text=""
            
            no_history += 1 
            print(f"ImpressionID {impression} has no click history.")


        data_dict[impression] = {key : behaviors.loc[impression][key] for key in behaviors}

        data_dict[impression]["title_emb"] = vectorizer.transform([title_text]).toarray()[0]
        data_dict[impression]["abstract_emb"] = vectorizer.transform([abstract_text]).toarray()[0]
        data_dict[impression]["abstract_title_emb"] = vectorizer.transform([title_abstract_text]).toarray()[0]

    return data_dict, no_history


def format_impressions(behaviors, news, vectorizer, k_samples = 7):
    """
    Format the behaviors files so that each impression is a row

    Args:
        behaviors ([DataFrame]): behaviors.tsv file. Will format data in "impressions" column.
        k_samples (int, optional): Number of negative samples of impression log to add, as there can be quite a few (simple "negative sampling"). Defaults to 7.

    Returns:
        [dict]: dictionary with new format and impression article embedding
    """

    data_dict = {}
    sample_no = 0        

    for index in behaviors.index:
        
        samples = behaviors.loc[index]["impressions"].split(" ")

        for i, sample in enumerate(samples): 
            
            if "-1" in sample or i < k_samples:
                temp = sample.split("-")

                # Temporal
                temp_title = str(news.loc[temp[0]]["title"])
                temp_abstract = str(news.loc[temp[0]]["abstract"])
                temp_title_abstract = temp_title + " " + temp_abstract

                target_title_emb = vectorizer.transform([temp_title]).toarray()[0]
                target_abstract_emb = vectorizer.transform([temp_abstract]).toarray()[0]
                target_title_abstract_emb = vectorizer.transform([temp_title_abstract]).toarray()[0]

                # Copy existings column names:
                data_dict[sample_no] = {key : behaviors.loc[index][key] for key in behaviors}

                # Add new column names:
                data_dict[sample_no]["article_id"] = temp[0]
                data_dict[sample_no]["target_title_emb"] = target_title_emb
                data_dict[sample_no]["target_abstract_emb"] = target_abstract_emb
                data_dict[sample_no]["target_title_abstract_emb"] = target_title_abstract_emb
                data_dict[sample_no]["article_target"] = temp
                data_dict[sample_no]["target"] = int(temp[1])
                
                sample_no += 1

    return data_dict