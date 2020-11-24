import json

from umap import UMAP
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataset.adult import Adult

class VisualState:

    def __init__(self, data):
        self.data = data
        self.umap_2d = UMAP(random_state=0)

    def get_point_id(self, interaction_data):
        curve_number = interaction_data['curveNumber']
        label = float(self.fig['data'][curve_number]['name'])
        return np.where(self.data.y == label)[0][interaction_data['pointNumber']]

    def create_fig(self):
        self.projections = self.umap_2d.fit_transform(self.data.X)
        self.fig = make_subplots(rows=1, cols=1)
        for label in np.unique(self.data.y):
            where = self.data.y == label
            self.fig.add_trace(
                go.Scatter(x=self.projections[where, 0], y=self.projections[where, 1],
                        name=str(label), mode='markers'
                ), row=1, col=1)
        self.fig.update_layout(clickmode='event+select')
        return self.fig

    def update_selected_data(self, selected_data):
        if selected_data is not None:
            rep = {}
            for point in selected_data['points']:
                point_id = int(self.get_point_id(point))
                rep[point_id] = self.data.as_json(point_id)
            return json.dumps(rep, indent=4)
        return json.dumps(None)

class Visualization(dash.Dash):

    def __init__(self, data):
        super().__init__(__name__)#, external_stylesheets=external_stylesheets)
        self.state = VisualState(data)
        self._setup_page()
        self.callback(
            Output('data-info', "children"),
            [Input("figure", "selectedData")]) (self.state.update_selected_data)

    def _setup_page(self):
        self.layout = html.Div([
            # PLOTS
            dcc.Graph(
                id='figure',
                figure=self.state.create_fig()
            ),
            html.Div([
                dcc.Markdown("""
                    **Selected Data**
                """),
                html.Pre(id='data-info')
            ])
        ])

if __name__ == "__main__":
    data = Adult('', subset_size=500)
    app = Visualization(data)
    app.run_server(debug=False)