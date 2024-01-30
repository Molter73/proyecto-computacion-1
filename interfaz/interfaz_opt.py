from dash import Dash, html, dcc, dash_table, Output, Input, State, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import requests

# Inicializamos la aplicación
application = Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;700;900&display=swap'
])

# Ajustes responsivos para la disposición de elementos
application.layout = html.Div(className='background', children=[
    html.Div(className='container', children=[
        
        # Logo
        html.Div(className='row', children=[
            html.Div(className='twelve columns', style={'textAlign': 'center'}, children=[
                html.Img(src='/assets/img/logo-uni.png',
                         style={'maxWidth': '150px', 'marginBottom': '20px'})
            ])
        ]),
        
        # Título
        html.Div(className='row', children=[
            html.Div(className='twelve columns', children=[
                html.H1(
                    'Inferencia de texto generado por máquina (MGT)',
                    className='text',
                    style={
                        'textAlign': 'center',
                        'fontFamily': 'Red Hat Display',
                        'fontWeight': 'bold',
                        'padding': '10px 10px'
                    }
                ),
                html.Hr(style={'borderTop': '3px solid #000',
                        'margin': '20px 0'}) 
            ])
        ]),

        # Selector de modelo y área de texto
        html.Div(className='row', children=[
            # Selector de modelo
            html.Div(className='six columns', children=[
                html.H3(
                    'Selecciona un modelo',
                    className='text',
                    style={'textAlign': 'center'}
                ),
                dcc.RadioItems(
                    options=[
                        {'label': ' Identificación', 'value': 'identification'},
                        {'label': ' Atribución', 'value': 'attribution'},
                    ],
                    id='seleccion_modelo',
                    labelStyle={'display': 'block', 'margin': '5px'},
                    style={'textAlign': 'center', 'fontSize': '18px'}
                ),
            ]),
            # Área de texto
            html.Div(className='six columns', children=[
                html.H3(
                    'Introduce tu texto',
                    className='text',
                    style={'textAlign': 'center'}
                ),
                dcc.Textarea(id='txt_area', value='', className='textarea'),
                html.Button('Analizar texto', id='button_txt_area',
                            n_clicks=0, className='button'),
            ])
        ]),

        # Resultados
        html.Div(className='row', children=[
            html.Div(className='twelve columns', children=[
                html.Div(id='resultados_metricas', style={'margin': '20px'}),
            ])
        ]),
        
        # Footer
        html.Div(className='row', children=[
            html.Div(className='twelve columns', style={
                'textAlign': 'center',
                'padding': '20px 0',
                'backgroundColor': '#e94619',
                'color': 'white',
                'fontSize': '14px',
                'marginTop': '30px',
                'bottom': '0',
                'width': '100%'
            }, children=[
                html.P(
                    'El proyecto ha sido realizado por Alejandro Delgado, Alzaro Álvarez, Brenda Solórzano y Mauro Moltrasio. 2024.')
            ])
        ])
    ])
])

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
