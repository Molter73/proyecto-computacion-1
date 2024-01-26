from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import requests

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)
font = "https://fonts.googleapis.com/css2?family=PT+Serif:wght@700&family=Source+Serif+4:opsz@8..60&display=swap"
# Inicializamos la aplicación
application = Dash(__name__)

application.layout = html.Div(
    [
        html.Link(
            href=font,
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
            [html.H3(className="titulo_opciones",
                     children="Selecciona un modelo")]
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
        html.Div([html.H3(className="titulo_opciones",
                 children="Introduce tu texto")]),
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
    State("seleccion_modelo", component_property="value"),
    State("txt_area", "value"),
)
def odel_selection(n_clicks, classification, text):
    if classification is None or text is None:
        raise PreventUpdate

    response = requests.post("http://127.0.0.1:5000", json={
        "text": text,
        "classification": classification
    })

    if not response.ok:
        return html.P(response.json()["error"], className="res")

    data = [
        {
            "Modelo": name,
            "Label": values["label"],
            "Probabilidad": values["proba"] if values["proba"] is not None else "NA",
        } for name, values in response.json()["predictions"].items()
    ]

    return dash_table.DataTable(
        data=data,
        page_size=12,
        style_table={"overflowX": "auto"},
    )


if __name__ == "__main__":
    application.run(
        debug=True
    )  # La aplicación se reiniciará automáticamente cada vez que se haga un cambio.
