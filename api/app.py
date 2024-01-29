from flask import Blueprint, request, abort, jsonify, current_app as app


classifiers = Blueprint('classifiers', __name__)


@classifiers.route("/", methods=["POST"])
def main():
    data = request.get_json()
    text = data["text"]
    classification = data.get("classification", "")

    if classification == "identification":
        return identification(text)
    elif classification == "attribution":
        return attribution(text)
    abort(400, description="Bad Request")


@classifiers.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


def proba(pipeline, X, y):
    if not hasattr(pipeline, "predict_proba"):
        return None
    return pipeline.predict_proba(X)[0][y]


def get_gpu_prediction(pipeline, text):
    prediction = pipeline(text)[0][0]
    return {
        "label": prediction["label"],
        "proba": prediction["score"],
    }


def get_predictions(models: dict, text):
    predictions = {}

    if "CPU" in models.keys():
        m = models["CPU"]
        labels = m["labels"]
        vectorizer = m['vectorizer']
        X = vectorizer.transform([text])

        for name, pipeline in m["models"].items():
            y = pipeline.predict(X)[0]
            predictions[name] = {
                "label": labels[y],
                "proba": proba(pipeline, X, y),
            }

    if "GPU" in models.keys():
        m = models["GPU"]
        for name, pipeline in m.items():
            predictions[name] = get_gpu_prediction(pipeline, text)

    return predictions


def identification(text: str):
    models = app.models["identification"]
    return {
        "task": "identification",
        "predictions": get_predictions(models, text),
    }


def attribution(text: str):
    models = app.models["attribution"]
    return {
        "task": "attribution",
        "predictions": get_predictions(models, text),
    }
