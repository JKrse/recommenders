# ========================================================================

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from sklearn.preprocessing import LabelEncoder, StandardScaler

import time
# ========================================================================


def LabelEncode_sparse_features(data, sparse_features):

    for feat in sparse_features:
        lbe = LabelEncoder() 
        data[feat] = lbe.fit_transform(data[feat])
    
    return data


def StandardScore_dense_features(data, dense_features):

    scaler = StandardScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    return data


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


def __missing_values(data, sparse_col, dense_col):
    
    data[sparse_col] = data[sparse_col].fillna('-1', )
    data[dense_col] = data[dense_col].fillna(0, )
    return data


class timeWrapper:

    def TimeStart():
        return time.time()

    def TimeEnd(first_timer):
        return time.time() - first_timer

    def print_TimeTaken(time_taken):
        print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(time_taken)))
        return time.strftime("%H:%M:%S",time.gmtime(time_taken))





