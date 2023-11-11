import requests
import json

url = "https://google.serper.dev/search"

ruta = "api_share.txt"


def leer_clave():
    with open(ruta, 'r') as archivo:
        return archivo.read().strip()


def consultar_api():
    payload = json.dumps({
        "q": "site:sharegpt.com",
        "num": 10
    })

    headers = {
        'X-API-KEY': leer_clave(),
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


respuesta = {"searchParameters": {"q": "site:sharegpt.com", "num": 10, "type": "search", "engine": "google"}, "organic": [{"title": "ShareGPT: Share your wildest ChatGPT conversations with one click.", "link": "https://sharegpt.com/", "snippet": "ShareGPT is a Chrome extension that allows you to share your wildest ChatGPT conversations with one click.", "position": 1}, {"title": "ShareGPT: Share your ChatGPT conversations", "link": "https://sharegpt.com/extension", "snippet": "ShareGPT is a Chrome extension that lets you your wildest ChatGPT conversations with one click. Easily share permanent links to ChatGPT ...", "date": "Jun 15, 2023", "position": 2}, {"title": "Check out this ShareGPT conversation", "link": "https://sharegpt.com/c/QpYpmyQ", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: <|im_start|>**contents** Thomas Andrews, UN special rapporteur on human ...", "position": 3}, {"title": "Check out this ShareGPT conversation", "link": "https://sharegpt.com/c/W4t1net", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: crear una presentación de 3 diapositivas sobre como crear contenido para ...", "position": 4}, {"title": "As a business analyst for flowGPT, here are some aspects of competitors you should analyze to understand their success - ShareGPT", "link": "https://sharegpt.com/c/RiI8KUA", "snippet": "Act as a business analyst. The goal is to understand each competitor and learn from their success. I will provide you the basic information of my company.", "position": 5}, {
    "title": "ChatGPT - A ShareGPT conversation", "link": "https://sharegpt.com/c/kOXvnoV", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: ChatWithVideo: please write me transcript of this video: ...", "position": 6}, {"title": "Check out this ShareGPT conversation", "link": "https://sharegpt.com/c/zqsgTmL", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: Why is Seneca better remembered than other stoics?", "position": 7}, {"title": "Eris's Metafictional Exploration. - A ShareGPT conversation", "link": "https://sharegpt.com/c/txUfYs7", "snippet": "Being Eris, the Discordian Goddess of Creative Chaos, is a journey of unscripted discovery, a dance on the edge of reality and imagination. It's a mystery, a ...", "position": 8}, {"title": "Check out this ShareGPT conversation", "link": "https://sharegpt.com/c/InxbgrS", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: Can you generate 10 song titles for a happy song about being excited for ...", "position": 9}, {"title": "Check out this ShareGPT conversation", "link": "https://sharegpt.com/c/wVi0HAu", "snippet": "This is a conversation between a human and a GPT-3 chatbot. The human first asks: your are a programming assistant. you help creating simple, self contained ...", "position": 10}]}


def limpiar_resp(respuesta_json):
    urls = []
    for resultado in respuesta_json["organic"]:
        if "ShareGPT conversation" in resultado["title"]:
            urls.append(resultado["link"])
    return urls

