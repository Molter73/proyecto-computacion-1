import requests
import json

url = "https://google.serper.dev/search"

ruta_api_key = "api_share.txt"


def leer_clave():
    with open(ruta_api_key, 'r') as archivo:
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

    return (response.json)


def limpiar_resp(respuesta_json):
    urls = []
    for resultado in respuesta_json["organic"]:
        if "ShareGPT conversation" in resultado["title"]:
            urls.append(resultado["link"])
    return urls


def main():
    respuesta = consultar_api()
    print(limpiar_resp(respuesta))


if __name__ == "__main__":
    main()
