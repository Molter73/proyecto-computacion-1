import os
import pickle

MODELS = {
    "identification": {},
    "attribution": {},
}
identification_pickle = "models/modelsA.pkl"
attribution_pickle = "models/modelsB.pkl"

if os.path.isfile(identification_pickle):
    print("identification models")
    with open(identification_pickle, "rb") as f:
        MODELS["identification"]["CPU"] = pickle.load(f)

if os.path.isfile(attribution_pickle):
    print("attribution models")
    with open(attribution_pickle, "rb") as f:
        MODELS["attribution"]["CPU"] = pickle.load(f)
