# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"
MLFLOW_DIR = "mlruns"

# data
TARGET_COL = "rating"
ITEM_IDX_COL = "movie_rn"
ITEM_ID_COL = "movie_id"
ITEM_TEXT_COL = "movie_text"
USER_IDX_COL = "user_rn"
USER_ID_COL = "user_id"
USER_TEXT_COL = "user_text"

# model
TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 2**5
PADDING_IDX = 0
METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}
TOP_K = 20

# serving
CHECKPOINT_PATH = "checkpoint.ckpt"
ITEMS_TABLE_NAME = "movies"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "mf_torch"
PROCESSORS_JSON = "processors.json"
TRANSFORMER_PATH = "transformer"
USERS_TABLE_NAME = "users"
