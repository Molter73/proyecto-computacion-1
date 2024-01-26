from dash import Dash, html, dcc, dash_table, Output, Input, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import json

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)
# Inicializamos la aplicación
aplication = Dash(__name__)

aplication.layout = html.Div(
    [
        html.Link(
            href="https://fonts.googleapis.com/css2?family=PT+Serif:wght@700&family=Source+Serif+4:opsz@8..60&display=swap",
            rel="stylesheet",
        ),
        html.Div(
            [
                html.H1(
                    className="titulo_app",
                    children="Inferencia de texto generado por máquina (MGT)",
                )
            ]
        ),
        html.Div(
            [html.H3(className="titulo_opciones", children="Selecciona un modelo")]
        ),
        html.Div(
            className="opciones",
            children=dcc.RadioItems(
                options=[
                    {
                        "label": "Identificación",
                        "value": "identification",
                    },
                    {
                        "label": "Atribución",
                        "value": "attribution",
                    },
                ],
                id="seleccion_modelo",
            ),
        ),
        html.Div([html.H3(className="titulo_opciones", children="Introduce tu texto")]),
        html.Div([dcc.Textarea(className="txt_area", id="txt_area", value="")]),
        html.Button(
            className="boton_txt_analizar",
            id="button_txt_area",
            children="Analizar texto",
        ),
        html.Div(
            [
                html.H3(
                    className="titulo_opciones",
                    children="Resultados de la clasificación MGT",
                ),
                html.Div(id="titulo_res_metricas"),
                html.Div(id="resultados_metricas"),
                html.P("Task:", className="res", id="p_task"),
                html.P("Label:", className="res", id="p_label"),
                html.P("Prob:", className="res", id="p_prob"),
            ],
        ),
        html.Div(
            [
                html.H3(
                    className="titulo_opciones",
                    children="Distribución de probabilidad MGT",
                )
            ]
        ),
        html.Div(
            className="fila",
            children=[
                html.Div(
                    className="columnas",
                    children=[
                        dash_table.DataTable(
                            data=df.to_dict("records"),
                            page_size=12,
                            style_table={"overflowX": "auto"},
                        )
                    ],
                ),
                html.Div(
                    className="columnas", children=[dcc.Graph(figure={}, id="datos")]
                ),
            ],
        ),
    ]
)


def graphic_features(op_elegida):
    grafica = px.histogram(
        df, x="continent", y=op_elegida, histfunc="avg", color="green"
    )
    return grafica


# Se llamará a esta función cada vez que se cambie el valor de la entrada para actualizar la salida.
@callback(
    Output("resultados_metricas", component_property="children"),
    Input("button_txt_area", component_property="n_clicks"),
    [State("seleccion_modelo", component_property="value"), State("txt_area", "value")],
)
def model_selection(text, classification):
    data = {"text": text, "classification": classification}
    data_to_json = json.dumps(data)

    response = requests.post("http://127.0.0.1:5000", data=data_to_json)
    if response.status_code == 200:
        print("Solicitud exitosa")
        print(response.json())
    else:
        print("Solicitud fallida")

    return {
        "text": text,
        "classification": classification,
    }


def get_metrics(task, label, prob):
    return {
        "task": task,
        "label": label,
        "prob": prob,
    }


if __name__ == "__main__":
    aplication.run(
        debug=True
    )  # La aplicación se reiniciará automáticamente cada vez que se haga un cambio.
