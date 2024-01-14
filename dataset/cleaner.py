import argparse
import sys

from langdetect import detect
import pandas as pd


def clean(df):
    df.drop_duplicates(subset='Text', inplace=True)

    filter_length = df['Text'].str.len() > 20
    df = df[filter_length]
    filter_lang = df['Text'].apply(lambda x: detect(x) == "en")
    df = df[filter_lang]
    return df


def exportar_archivo(df, archivo_tsv):
    df.reset_index(drop=True, inplace=True)
    df['ID'] = df.index
    df.to_csv(archivo_tsv, sep='\t', index=False)


def main(input, output):
    df = clean(pd.read_csv(input, sep='\t'))
    exportar_archivo(df, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='path a un archivo json con los textos a limpiar')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='path al archivo de salida')

    args = parser.parse_args()
    input = open(args.input, 'r')
    output = sys.stdout if args.output is None else open(args.output, 'w')
    main(input, output)
