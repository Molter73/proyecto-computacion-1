from langdetect import detect 
from scrapper import main as scrap
import pandas as pd

def clean(elementos, label):
    df = pd.DataFrame(elementos, columns=['Text'])
    df['Label'] = label
    df.drop_duplicates(subset='Text', inplace=True)

    filter_df = df[df['Text'].str.len() > 20] 
    filter_lan = df[filter_df['Text'].apply(lambda x: detect(x) == "en")]  
    return filter_lan

def exportar_archivo(df, archivo_tsv):
    df.reset_index(drop=True, inplace=True)    
    df['ID'] = df.index
    df.to_csv(archivo_tsv, sep='\t', index=False) 
    print(f'Se exportó el archivo: {archivo_tsv}')

def main():
    generado , humano = scrap()

    df_generado = clean(generado, 'IA')
    df_humano  = clean(humano, 'HUMANO')
    df_total = pd.concat([df_generado, df_humano])

    archivo_csv = 'ejemplo.csv'
    #archivo_tsv = 'ejemplo.tsv
    #print(f'Se ha creado el archivo TSV: {archivo_tsv}')

    exportar_archivo(df_total, archivo_csv)

main()    
#if __name__ == '__main__':
#    main()














