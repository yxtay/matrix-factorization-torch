# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"
MLFLOW_DIR = "mlruns"

# data
TARGET_COL = "rating"
USER_RN_COL = "user_rn"
USER_ID_COL = "user_id"
USER_FEATURE_NAMES = {
    "user_id": "user_id",
    "gender": "gender",
    "age": "age",
    "occupation": "occupation",
    "zipcode": "zipcode",
}
ITEM_RN_COL = "movie_rn"
ITEM_ID_COL = "movie_id"
ITEM_FEATURE_NAMES = {
    "movie_id": "movie_id",
    "genres": "genres",
}

# model
BATCH_SIZE = 2**8
NUM_HASHES = 2
NUM_EMBEDDINGS = 2**16 + 1
EMBEDDING_DIM = 32
PADDING_IDX = 0
METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}
TOP_K = 20

# serving
CHECKPOINT_PATH = "checkpoint.ckpt"
EXPORTED_PROGRAM_PATH = "exported_program.pt"
ITEMS_TABLE_NAME = "movies"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "mf_torch"
SCRIPT_MODULE_PATH = "scriptmodule.pt"
USERS_TABLE_NAME = "users"
