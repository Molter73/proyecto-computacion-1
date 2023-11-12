import requests
from bs4 import BeautifulSoup

urls_prueba = ['https://sharegpt.com/c/W4t1net',
               'https://sharegpt.com/c/kOXvnoV',
               'https://sharegpt.com/c/zqsgTmL',
               'https://sharegpt.com/c/txUfYs7',
               'https://sharegpt.com/c/InxbgrS',
               'https://sharegpt.com/c/wVi0HAu',
               'https://sharegpt.com/c/QpYpmyQ'
               ]


def consultar(url):
    respuesta = requests.get(url)
    return BeautifulSoup(respuesta.content, 'html.parser')


def extraer_texto_ia(soup):
    result = []
    for s in soup:
        result.append(
            ' '.join([
                limpiar(text.text) for text in s.find_all(['p', 'h1', 'h2', 'h3', 'b', 'p', 'a', 'li'])
            ]))

    return result


def extraer_texto_humano(soup):
    return [
        ' '.join([
             limpiar(s.text) for s in soup.find_all('p', class_='pb-2 whitespace-prewrap')
        ])
    ]


def extraer_texto(soup):
    texto_generado = extraer_texto_ia(soup.find_all('div', class_='utils_response__b5jEi'))
    texto_humano = extraer_texto_humano(soup)
    return texto_generado, texto_humano


def limpiar(texto):
    info_saltos = texto.replace('\n', ' ')
    info_tab = info_saltos.replace('\t', ' ')
    return info_tab


def main():
    generado = []
    humano = []

    for url in urls_prueba:
        soup = consultar(url)
        g, h = extraer_texto(soup)
        generado.extend(g)
        humano.extend(h)

    print(f'Texto máquina: {generado}')
    print(f'Texto humano: {humano}')


if __name__ == '__main__':
    main()
