from deepctr.models import NFM, DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


def wide_deep_inputs(df_data, sparse_feature_col, dense_feature_col, model_type="DeepFM"): 

    sparse_embedding = [SparseFeat(feat, vocabulary_size=df_data[feat].max() + 1, embedding_dim=4) for feat in sparse_feature_col]
    dense_embedding  = [DenseFeat(feat, 1,) for feat in dense_feature_col]

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



def __missing_values(data, sparse_col, dense_col):
    # If missing values: 
    data[sparse_col] = data[sparse_col].fillna('-1', )
    data[dense_col] = data[dense_col].fillna(0, )
    return data


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