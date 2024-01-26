import argparse
import sys
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from crawler import crawl, leer_clave
from scrapper import scrap
from cleaner import clean, exportar_archivo


def split_dataset(output, df):
    if output is None:
        return

    max_instances_per_class = 1500
    random_seed = 777  # set random seed for reproducibility
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=random_seed)
    train_df = train_df.groupby("label").sample(
        n=max_instances_per_class, random_state=random_seed)

    path = Path(output)
    dir = path.parent
    filename = path.stem
    ext = path.suffix

    with open(os.path.join(dir, f'{filename}-train{ext}'), 'w') as f:
        train_df.to_json(f, lines=True, orient='records')

    with open(os.path.join(dir, f'{filename}-test{ext}'), 'w') as f:
        test_df.to_json(f, lines=True, orient='records')


def main(api_key, output, count):
    print("Crawling ShareGPT...")
    urls = crawl(api_key, count)

    print("Scrapping ShareGPT...")
    df = pd.DataFrame()
    for url in urls:
        df = pd.concat([df, scrap(url)])

    print("Limpiando y exportando resultados...")
    df = clean(df)
    output_fd = sys.stdout if output is None else open(args.output, 'w')
    exportar_archivo(df, output_fd)

    print("Particionando dataset...")
    split_dataset(output, df)


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
    output = args.output
    main(api_key, output, args.count)
