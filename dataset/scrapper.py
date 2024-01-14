import argparse
import requests
import sys

from bs4 import BeautifulSoup
import pandas as pd


def consultar(url):
    respuesta = requests.get(url.strip())
    respuesta.raise_for_status()
    return BeautifulSoup(respuesta.content, 'html.parser')


def extraer_texto_ia(soup):
    result = []
    for s in soup:
        result.append(
            ' '.join([
                limpiar(text.text) for text in s.find_all(['p', 'h1', 'h2', 'h3', 'b', 'p', 'a', 'li'])
            ]))

    result = pd.DataFrame(result, columns=['Text'])
    result['Label'] = 'IA'
    return result


def extraer_texto_humano(soup):
    result = [
        limpiar(s.text) for s in soup.find_all('p', class_='pb-2 whitespace-prewrap')
    ]

    result = pd.DataFrame(result, columns=['Text'])
    result['Label'] = 'HUMANO'
    return result


def extraer_texto(soup):
    texto_generado = extraer_texto_ia(
        soup.find_all('div', class_='utils_response__b5jEi'))
    texto_humano = extraer_texto_humano(soup)

    df = pd.concat([texto_generado, texto_humano])
    return df


def limpiar(texto):
    info_saltos = texto.replace('\n', ' ')
    info_tab = info_saltos.replace('\t', ' ')
    return info_tab


def scrap(url):
    soup = consultar(url)
    return extraer_texto(soup)


def main(input, output):
    df = pd.DataFrame()

    for url in input.readlines():
        df = pd.concat([df, scrap(url)])

    df.to_csv(output, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default=None,
                        help='path a un archivo con las URL a scrappear.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='path al archivo de salida a generar')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')
    with open(args.input, 'r') as input:
        main(input, output)
