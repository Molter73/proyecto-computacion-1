import os
import pickle

MODELS = {
    "identification": {},
    "attribution": {},
}

models_subdir = os.environ.get("API_MODELS", "models")

identification_pickle = os.path.join(models_subdir, "modelsA.pkl")
attribution_pickle = os.path.join(models_subdir, "modelsB.pkl")

if os.path.isfile(identification_pickle):
    with open(identification_pickle, "rb") as f:
        MODELS["identification"]["CPU"] = pickle.load(f)

if os.path.isfile(attribution_pickle):
    with open(attribution_pickle, "rb") as f:
        MODELS["attribution"]["CPU"] = pickle.load(f)
