from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import requests

# Inicializamos la aplicación
application = Dash(__name__)

application.layout = html.Div(
    [
        html.Link(
            href="https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;700;@900&display=swap",
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
        html.Div(className="separador"),
        html.Div(
            [html.H3(className="titulo_opciones", children="Selecciona un modelo")]
        ),
        html.Div(
            [
                html.H5(
                    children="Get the best results in text prediction.",
                    className="txt_d1",
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
            ],
            className="opt_1",
        ),
        html.Div(className="espacios"),
        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            className="titulo_opciones", children="Introduce tu texto"
                        )
                    ]
                ),
                html.Div([dcc.Textarea(className="txt_area", id="txt_area", value="")]),
                html.Button(
                    className="boton_txt_analizar",
                    id="button_txt_area",
                    children="Analizar texto",
                ),
            ],
            className="opt_2",
        ),
        html.Div(
            [
                html.H3(
                    className="titulo_opciones",
                    children="MGT classification results",
                ),
                html.Div(id="titulo_res_metricas"),
                html.Div(id="resultados_metricas"),
            ],
            className="opti",
        ),
    ]
)


# Se llamará a esta función cada vez que se cambie el valor de la entrada para actualizar la salida.
@callback(
    Output("resultados_metricas", component_property="children"),
    Input("button_txt_area", component_property="n_clicks"),
    State("seleccion_modelo", component_property="value"),
    State("txt_area", "value"),
)
def model_selection(n_clicks, classification, text):
    if classification is None or text is None:
        raise PreventUpdate

    response = requests.post(
        "http://127.0.0.1:5000", json={"text": text, "classification": classification}
    )

    if not response.ok:
        return html.P(response.json()["error"], className="res")

    data = [
        {
            "Modelo": name,
            "Label": values["label"],
            "Probabilidad": values["proba"] if values["proba"] is not None else "NA",
        }
        for name, values in response.json()["predictions"].items()
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
