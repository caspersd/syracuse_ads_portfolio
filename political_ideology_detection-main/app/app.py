import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import pickle
from rank_ideology import rank_politicians

# Initialize Dash app
app = dash.Dash(__name__)

# Default rankings
ranked_df = rank_politicians()
ranked_df = ranked_df.sort_values(by="Magnitude", ascending=True)
print(ranked_df)
options = [{'label': senator, 'value': senator} for senator in ranked_df['Senator']]

# Layout
app.layout = html.Div([
    html.H1("Political Ideology Ranking"),
    
    html.Label("Base Politician 1"),
    dcc.Dropdown(id='base1', options=options, value="Sen. Dan Sullivan", clearable=False),
    
    html.Label("Base Politician 2"),
    dcc.Dropdown(id='base2', options=options, value="Sen. Chuck Schumer", clearable=False),
    
    dcc.Graph(id='ranking-bar-plot'),
])

# Callback to update graph
@app.callback(
    Output('ranking-bar-plot', 'figure'),
    [Input('base1', 'value'), Input('base2', 'value')]
)
def update_plot(base1, base2):
    ranked_df = rank_politicians(base1, base2)
    ranked_df = ranked_df.sort_values(by="Magnitude", ascending=True)
    fig = px.bar(ranked_df, x='Senator', y='Magnitude', title='Political Ideology Ranking')
    return fig

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
