from flask import Flask, request, abort, jsonify

app = Flask(__name__)


@app.route("/", methods=["POST"])
def main():
    data = request.get_json()
    text = data["text"]
    classification = data.get("classification", "identification")

    if classification == "identification":
        return identification(text)
    elif classification == "attribution":
        return attribution(text)
    abort(400, description="Bad Request")


@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400


def make_response(task: str, label: str, prob: float):
    return {
        "task": task,
        "label": label,
        "prob": prob,
    }


def identification(text: str):
    return make_response("identification", "HUMANO", 0.75)


def attribution(text: str):
    return make_response("attribution", "chatGPT", 0.8)
