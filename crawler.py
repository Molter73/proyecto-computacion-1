import argparse
import sys
import os
import requests
import json

url = "https://google.serper.dev/search"


def leer_clave(api_key_file):
    if api_key_file is None:
        return os.getenv('SERPER_API_KEY')

    with open(api_key_file, 'r') as archivo:
        return archivo.read().strip()


def consultar_api(api_key, count, page):
    payload = json.dumps({
        "q": "site:sharegpt.com",
        "page": page,
        "num": count,
    })

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()

    return response.json()['organic']


def limpiar_resp(respuesta_json):
    urls = []
    for resultado in respuesta_json:
        if "ShareGPT conversation" in resultado["title"]:
            urls.append(resultado["link"])
    return urls


def crawl(api_key, count):
    page = 1
    respuesta = []
    while count > 0:
        respuesta.extend(consultar_api(api_key, count, page))
        count -= 100
        page += 1
    return limpiar_resp(respuesta)


def main(api_key, count, output):
    urls = crawl(api_key, count)
    output.write('\n'.join(urls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    api_key_help = 'path a un archivo con la clave de api serper (default: valor de variable de entorno SERPER_API_KEY)'
    parser.add_argument('--api-key-file', type=str, default=None,
                        help=api_key_help)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='path al archivo de salida a generar')
    parser.add_argument('-n', '--count', type=int, default=10,
                        help='cantidad de resultados a traer de serper')

    args = parser.parse_args()
    output = sys.stdout if args.output is None else open(args.output, 'w')
    api_key = leer_clave(args.api_key_file)
    if api_key is None:
        sys.exit('No se pudo obtener la clave de API')

    main(api_key, args.count, output)
