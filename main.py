import argparse
import sys

import pandas as pd

from crawler import crawl, leer_clave
from scrapper import scrap
from cleaner import clean, exportar_archivo


def main(api_key, output, count):
    print("Crawling ShareGPT...")
    urls = crawl(api_key, count)

    print("Scrapping ShareGPT...")
    generado = []
    humano = []
    for url in urls:
        g, h = scrap(url)
        generado.extend(g)
        humano.extend(h)

    print("Limpiando y exportando resultados...")
    df_generado = clean(generado, 'IA')
    df_humano = clean(humano, 'HUMANO')
    df = pd.concat([df_generado, df_humano])

    exportar_archivo(df, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    api_key_help = 'path a un archivo con la clave de api serper (default: valor de variable de entorno SERPER_API_KEY)'
    parser.add_argument('--api-key-file', type=str, default=None,
                        help=api_key_help)
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='path al archivo de salida a generar')
    parser.add_argument('-n', '--count', type=int, default=10,
                        help='cantidad de resultados a traer de serper')

    args = parser.parse_args()
    api_key = leer_clave(args.api_key_file)
    output = sys.stdout if args.output is None else open(args.output, 'w')
    main(api_key, output, args.count)
