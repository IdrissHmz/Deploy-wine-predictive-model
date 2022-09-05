import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist


app = FastAPI(title="Predicting Wine Class with batching")

# Represents a batch of wines
class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_items=13, max_items=13)]


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("../app/wine.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)


@app.get("/")
def home():
    return "API is working as expected. This new version allows for batching. Now head over to http://localhost:81/docs"


@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}


# docker build -t wine-api .
#
# docker run --rm -p 81:80 mlepc4w2-ugl:with-batch
#
# curl -X POST http://localhost:81/predict \
#     -d @./wine-examples/batch_1.json \
#     -H "Content-Type: application/json"
#
# docker-compose up

