from umap import UMAP
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# from plotly.colors import qualitative

from get_shap import datapoints_to_test, predictions

class Data:
    def __init__(self):
        self.x = datapoints_to_test
        self.y = predictions

class VisualState:

    def __init__(self, data):
        self.data = data

    def create_fig(self):
        self.umap_2d = UMAP(random_state=0)
        self.projections = self.umap_2d.transform(self.data.x)
        self.fig = make_subplots(rows=1, cols=1)
        for label in np.unique(self.data.y):
            where = self.data.y == label
            self.fig.add_trace(
                go.Scatter(x=self.projections[where, 0], y=self.projections[where, 1],
                        name=str(label), mode='markers'
                ), row=1, col=1)
        return fig

class Visualization(dash.Dash):

    def __init__(self, data):
        super().__init__(__name__)#, external_stylesheets=external_stylesheets)
        self.state = VisualState(data)
        self._setup_page()

    def _setup_page(self):
        self.layout = html.Div([
            # PLOTS
            dcc.Graph(
                id='basic-interactions',
                figure=self.state.create_fig()
            ),
        ])

if __name__ == "__main__":
    data = Data()
    app = Visualization(data)
    app.run_server(debug=False)