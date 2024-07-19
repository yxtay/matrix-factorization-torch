# data
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
USER_IDX = "user_idx"
USER_FEATURE_NAMES = ["user_id", "gender", "age", "occupation", "zipcode"]
ITEM_IDX = "movie_idx"
ITEM_FEATURE_NAMES = ["movie_id", "genres"]

# serving
EMBEDDER_PATH = "scripted_module.pt"
LANCE_DB_PATH = "lancedb"
MODEL_NAME = "mf-torch"
MOVIES_DOC_PATH = "movies"
MOVIES_TABLE_NAME = "movies"
