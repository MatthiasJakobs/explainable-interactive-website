import json

from umap import UMAP
import numpy as np
import torch

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# from plotly.colors import qualitative

from dataset.adult import Adult
from model import Net

class Data:
    def __init__(self):
        self.net = Net()
        self.net.load_state_dict(torch.load("model.pth"))

        a_train = Adult("dataset", train=True)
        a_test = Adult("dataset", train=False)

        self.train = torch.zeros(len(a_train), *a_train[0]['x'].shape)
        self.test = torch.zeros(len(a_test), *a_test[0]['x'].shape)
        self.labels_test = torch.zeros(len(a_test), *a_test[0]['y'].shape)

        # indices_1h = get_categorical_indices(a_train.X, categorical_columns)

        for i in range(len(a_train)):
            self.train[i] = a_train[i]['x']
        for i in range(len(a_test)):
            self.test[i] = a_test[i]['x']
            self.labels_test[i] = a_test[i]['y']

    def repr(self, point_id):
        return str(self.test[point_id])

class VisualState:

    def __init__(self, data):
        self.data = data
        self.predictions = self.data.labels_test
        self.umap_2d = UMAP(random_state=0)

    def get_point_id(self, interaction_data):
        curve_number = interaction_data['curveNumber']
        label = float(self.fig['data'][curve_number]['name'])
        return np.where(self.predictions == label)[0][interaction_data['pointNumber']]

    def create_fig(self):
        self.projections = self.umap_2d.fit_transform(self.data.test)
        self.fig = make_subplots(rows=1, cols=1)
        for label in np.unique(self.predictions):
            where = np.random.choice(np.where(self.predictions == label)[0], 1000)
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
                rep[point_id] = self.data.repr(point_id)
            return json.dumps(rep)
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
                html.Div(id='data-info')
            ])
        ])

if __name__ == "__main__":
    data = Data()
    app = Visualization(data)
    app.run_server(debug=False)