import os
from pathlib import Path
from typing import List
import pandas as pd
from pandas.core.frame import DataFrame

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources 
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set


class utils_MIND: 
    """Utility function to handle MIND. 
    Contain function to download the dataset and to load the files.
    """

    def __init__(self) -> None:
        pass
    

    def download_MIND_data(self, MIND_type: str="demo", data_path: str=None) -> None:
        """Load Data for Modelling
        Loads relevant data set for modelling. If data is not downloaded
        already, it will be downloaded.
        Args:
            MIND_type (str, optional): Which MIND dataset to use. 
            Defaults to "demo". Choose from 'demo', 'small' and 'large'.
            data_path (str, optional): Where to store data. 
            Defaults to None.
        """
   
        print("Getting data for modelling...")

        #### download and load data
        if data_path is None:      
            data_path = os.path.join(Path.home(), ".MIND")

        self.train_news_file = os.path.join(data_path, 'train', r'news.tsv')
        self.train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
        self.valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
        self.valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
        self.wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
        self.userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
        self.wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")

        self.mind_url, self.mind_train_dataset, self.mind_dev_dataset, self.mind_utils = get_mind_data_set(MIND_type)

        if not os.path.exists(self.train_news_file):
            download_deeprec_resources(self.mind_url, os.path.join(data_path, 'train'), self.mind_train_dataset)

        if not os.path.exists(self.valid_news_file):
            download_deeprec_resources(self.mind_url, \
                                    os.path.join(data_path, 'valid'), self.mind_dev_dataset)

        print("Datasets downloaded.")
    

    def load_behaviors(self, data_path: str) -> DataFrame:
        # The behaviors.tsv file contains the impression logs and users' news click histories. 
        # It has 5 columns divided by the tab symbol:
        # - Impression ID. The ID of an impression.
        # - User ID. The anonymous ID of a user.
        # - Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
        # - History. The news click history (ID list of clicked news) of this user before this impression.
        # - Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click).
        behaviors_path = os.path.join(data_path, 'behaviors.tsv')
        data = pd.read_table(behaviors_path, header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        return data


    def load_news(self, data_path) -> DataFrame:
        # The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file.
        # It has 7 columns, which are divided by the tab symbol:
        # - News ID
        # - Category
        # - Subcategory
        # - Title
        # - Abstract
        # - URL
        # - Title Entities (entities contained in the title of this news)
        # - Abstract Entities (entities contained in the abstract of this news)
        news_path = os.path.join(data_path, 'news.tsv')
        data = pd.read_table(news_path, header=None, names=[
                        'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                        'title_entities', 'abstract_entities'
                    ])
        return data


    def load_entity_embedding(self, data_path) -> DataFrame:
        # The entity_embedding.vec file contains the 100-dimensional embeddings
        # of the entities learned from the subgraph by TransE method.
        # The first column is the ID of entity, and the other columns are the embedding vector values.
        entity_embedding_path = os.path.join(data_path, 'entity_embedding.vec')
        entity_embedding = pd.read_table(entity_embedding_path, header=None)
        entity_embedding['vector'] = entity_embedding.iloc[:, 1:101].values.tolist()
        entity_embedding = entity_embedding[[0, 'vector']].rename(columns={0: "entity"})
        return entity_embedding


    def load_relation_embedding(self, data_path) -> DataFrame:
        # The relation_embedding.vec file contains the 100-dimensional embeddings
        # of the relations learned from the subgraph by TransE method.
        # The first column is the ID of relation, and the other columns are the embedding vector values.
        relation_embedding_path = os.path.join(data_path, 'relation_embedding.vec')
        relation_embedding = pd.read_table(relation_embedding_path, header=None)
        relation_embedding['vector'] = relation_embedding.iloc[:,
                                                            1:101].values.tolist()
        relation_embedding = relation_embedding[[0, 'vector']].rename(columns={0: "relation"})
        return relation_embedding
    
