import requests
from bs4 import BeautifulSoup

urls_prueba = ['https://sharegpt.com/c/W4t1net', 'https://sharegpt.com/c/kOXvnoV', 'https://sharegpt.com/c/zqsgTmL', 'https://sharegpt.com/c/txUfYs7', 'https://sharegpt.com/c/InxbgrS', 'https://sharegpt.com/c/wVi0HAu','https://sharegpt.com/c/QpYpmyQ']

def extraer_informacion(url):
    respuesta = requests.get(url)
    
    soup = BeautifulSoup(respuesta.content, 'html.parser')
    
    texto_generado = soup.find_all('div', class_='utils_response__b5jEi')
    for texto in texto_generado:
        for etiqueta in texto.find_all(['p', 'h1', 'h2', 'h3', 'b', 'p', 'a']):
            print("Texto generado por IA: ", etiqueta.get_text())

    texto_humano = soup.find_all('p', class_='pb-2 whitespace-prewrap')
    for texto in texto_humano:
        print("Texto generado por Humano: ", (texto.get_text()))

for urls in urls_prueba:
    info = extraer_informacion(urls)

def limpiar(texto):
    info_saltos = texto.replace('\n', ' ')
    info_tab = info_saltos.replace('\t', ' ')
    return info_tab

info_modificado = limpiar(info)