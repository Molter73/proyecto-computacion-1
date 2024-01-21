from dash import Dash, html, dcc, dash_table, Output, Input, callback
import plotly.express as px
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)
# Inicializamos la aplicación
aplicacion = Dash(__name__)

aplicacion.layout = html.Div(
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
            [html.H3(className="titulo_opciones", children="Seleccionar una tarea MGT")]
        ),
        html.Div(
            className="opciones",
            children=dcc.RadioItems(
                options=[
                    "Identificación de textos generados",
                    "Atribución de textos a modelos de generación",
                ],
                id="control_opciones",
            ),
        ),
        html.Div([html.H3(className="titulo_opciones", children="Introduce tu texto")]),
        html.Div([dcc.Textarea(className="txt_area", id="txt_area")]),
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
                )
            ]
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


# Se llamará a esta función cada vez que se cambie el valor de la entrada para actualizar la salida.
@callback(
    Output(component_id="datos", component_property="figure"),
    Input(component_id="control_opciones", component_property="value"),
)
def control_grafica(op_elegida):
    grafica = px.histogram(
        df, x="continent", y=op_elegida, histfunc="avg", color="green"
    )
    return grafica


if __name__ == "__main__":
    aplicacion.run(
        debug=True
    )  # La aplicación se reiniciará automáticamente cada vez que se haga un cambio.
