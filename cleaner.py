from langdetect import detect
from scrapper import main as scrap
import pandas as pd


def clean(elementos, label):
    df = pd.DataFrame(elementos, columns=['Text'])
    df['Label'] = label
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
    print(f'Se exportó el archivo: {archivo_tsv}')


def main():
    generado, humano = scrap()

    df_generado = clean(generado, 'IA')
    df_humano = clean(humano, 'HUMANO')
    df_total = pd.concat([df_generado, df_humano])

    archivo_tsv = 'ejemplo.tsv'

    exportar_archivo(df_total, archivo_tsv)


if __name__ == '__main__':
    main()
