import os
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


models_subdir = os.environ.get("API_MODELS", "models")

identification_models = {
    "pickle": os.path.join(models_subdir, "modelsA.pkl"),
    "roberta": os.path.join(models_subdir, "roberta-base", "subtaskA", "best"),
    "gbert": os.path.join(models_subdir, "google", "bert", "subtaskA", "best"),
    "id2label": {0: "human", 1: "machine"},
    "label2id": {"human": 0, "machine": 1},
}

attribution_models = {
    "pickle": os.path.join(models_subdir, "modelsB.pkl"),
    "roberta": os.path.join(models_subdir, "roberta-base", "subtaskB", "best"),
    "gbert": os.path.join(models_subdir, "google", "bert", "subtaskB", "best"),
    "id2label": {
        0: 'human',
        1: 'chatGPT',
        2: 'cohere',
        3: 'davinci',
        4: 'bloomz',
        5: 'dolly'
    },
    "label2id": {
        'human': 0,
        'chatGPT': 1,
        'cohere': 2,
        'davinci': 3,
        'bloomz': 4,
        'dolly': 5
    },
}


def load_gpu_model(path, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    return TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)


def load_models(models):
    res = {}
    gpu = {}

    pickle_file = models["pickle"]
    roberta_dir = models["roberta"]
    gbert_dir = models["gbert"]
    id2label = models["id2label"]
    label2id = models["label2id"]

    if os.path.isfile(pickle_file):
        print("Loading pickle")
        with open(pickle_file, "rb") as f:
            res["CPU"] = pickle.load(f)

    if os.path.isdir(roberta_dir):
        print("Loading roberta")
        gpu["roberta"] = load_gpu_model(roberta_dir, id2label, label2id)

    if os.path.isdir(gbert_dir):
        print("Loading gbert")
        gpu["gbert"] = load_gpu_model(roberta_dir, id2label, label2id)

    if gpu:
        res["GPU"] = gpu

    return res


MODELS = {
    "identification": load_models(identification_models),
    "attribution": load_models(attribution_models),
}

print("Initialization done")
