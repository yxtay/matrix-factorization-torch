# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"
TENSORBOARD_DIR = "lightning_logs"
MLFLOW_DIR = "mlruns"

# data
TARGET_COL = "rating"
ITEM_ID_COL = "movie_id"
ITEM_TEXT_COL = "movie_text"
ITEM_FEATURE_NAMES = ["title", "genres"]
USER_ID_COL = "user_id"
USER_TEXT_COL = "user_text"
USER_FEATURE_NAMES = ["gender", "age", "occupation", "zipcode"]

# model
ENCODER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 2**5
EMBEDDING_DIM = 384
PADDING_IDX = 0
METRIC = {"name": "val/RetrievalNormalizedDCG", "mode": "max"}
TOP_K = 20

# serving
CHECKPOINT_PATH = "checkpoint.ckpt"
PROCESSORS_JSON = "processors.json"
MODEL_PATH = "model"
EXPORTED_PROGRAM_PATH = "exported_program.pt"
ONNX_PROGRAM_PATH = "program.onnx"
ITEMS_TABLE_NAME = "movies"
LANCE_DB_PATH = "lance_db"
MODEL_NAME = "mf_torch"
SCRIPT_MODULE_PATH = "scriptmodule.pt"
USERS_TABLE_NAME = "users"
