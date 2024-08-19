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
METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}

# serving
CHECKPOINT_PATH = "checkpoint.ckpt"
ITEMS_TABLE_NAME = "movies"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "mf_torch"
SCRIPT_MODULE_PATH = "scriptmodule.pt"
EXPORTED_PROGRAM_PATH = "exported_program.pt"
