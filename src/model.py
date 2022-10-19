import pickle
from pathlib import Path
import numpy as np
import gensim.downloader as api
from utils import vectorize, project_data_preparation
import en_core_web_sm
from pydantic import BaseModel

nlp = en_core_web_sm.load()
WV = api.load('word2vec-google-news-300')

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


def predict_pipeline(text):
    data = project_data_preparation(text, WV)
    result = model.predict(np.reshape(data, (-1, data.shape[0])))
    return int(result)
