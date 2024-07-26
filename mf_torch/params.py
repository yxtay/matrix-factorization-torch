# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"
MLFLOW_DIR = "mlruns"

# data
USER_IDX = "user_id"
USER_FEATURE_NAMES = ["user_id", "gender", "age", "occupation", "zipcode"]
ITEM_IDX = "movie_id"
ITEM_FEATURE_NAMES = ["movie_id", "genres"]

# model
NUM_HASHES = 2
NUM_EMBEDDINGS = 2**16 + 1
EMBEDDING_DIM = 32
PADDING_IDX = 0

# serving
EMBEDDER_PATH = "scripted_module.pt"
LANCE_DB_PATH = "lancedb"
MODEL_NAME = "mf-torch"
ITEMS_DOC_PATH = "movies"
ITEMS_TABLE_NAME = "movies"
