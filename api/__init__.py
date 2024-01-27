import os
import pickle
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline


models_subdir = os.environ.get("API_MODELS", "models")

identification_models = {
    "CPU": {
        'path': os.path.join(models_subdir, "modelsA"),
        'labels': os.path.join(models_subdir, "labelsA.pkl"),
    },
    "roberta": os.path.join(models_subdir, "roberta-base", "subtaskA", "best"),
    "gbert": os.path.join(models_subdir, "google", "bert", "subtaskA", "best"),
    "id2label": {0: "human", 1: "machine"},
    "label2id": {"human": 0, "machine": 1},
}

attribution_models = {
    "CPU": {
        'path': os.path.join(models_subdir, "modelsB"),
        'labels': os.path.join(models_subdir, "labelsB.pkl"),
    },
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


def load_cpu_models(path, labels_file):
    if not os.path.isfile(labels_file):
        return {}

    cpu = {}
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)

    for file in [
        os.path.join(path, f) for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]:
        name = Path(file).stem
        print(f"Loading {name}")
        with open(file, "rb") as f:
            cpu[name] = pickle.load(f)

    if not cpu:
        return {}

    return {
        'models': cpu,
        'labels': labels,
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
    cpu = {}

    cpu_dir = models["CPU"]["path"]
    cpu_labels = models["CPU"]["labels"]
    roberta_dir = models["roberta"]
    gbert_dir = models["gbert"]
    id2label = models["id2label"]
    label2id = models["label2id"]

    if os.path.isdir(cpu_dir):
        cpu = load_cpu_models(cpu_dir, cpu_labels)

    if cpu:
        res["CPU"] = cpu

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
