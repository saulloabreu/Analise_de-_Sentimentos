from app import *
from dash_bootstrap_templates import ThemeSwitchAIO
import dash_bootstrap_components as dbc
import dash


app = dash.Dash(
    __name__, 
    external_stylesheets = [dbc.themes.DARKLY], 
    suppress_callback_exceptions = True, 
    meta_tags = [{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.config.suppress_callback_exceptions = True
app.scripts.config.serve_localy = True
app.title = 'Insights-Analise de Sentimentos'
server = app.server