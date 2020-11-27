import json

from umap import UMAP
import numpy as np
import torch

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataset.adult import Adult
from model import Net

class VisualState:

    def __init__(self, data):
        self.data = data
        self.umap_2d = UMAP(random_state=0)
        self.projections = self.umap_2d.fit_transform(self.data.numpy()[0])
        self.net = Net()
        self.pred = np.array(torch.argmax(self.net(self.data.torch()[0]), dim=-1))
        self.confuse = np.zeros_like(self.pred).astype(str)
        correct = self.data.numpy()[1] == self.pred
        self.confuse[correct] = 'CORRECT'
        self.confuse[np.invert(correct)] = 'WRONG'

    def get_point_id(self, interaction_data):
        curve_number = interaction_data['curveNumber']
        label = float(self.fig['data'][curve_number]['name'])
        return np.where(self.data.numpy()[1] == label)[0][interaction_data['pointNumber']]

    def update_figure(self, value, relayout_data):
        if 'PRED' in value:
            return self.create_fig(use_prediction=True, relayout_data=relayout_data)
        return self.create_fig(relayout_data=relayout_data)

    def create_fig(self, use_prediction=False, relayout_data=None):
        self.fig = make_subplots(rows=1, cols=1)
        if use_prediction:
            y = self.confuse
        else:
            y = self.data.numpy()[1]
        # plot traces
        for label in np.unique(y):
            where = y == label
            self.fig.add_trace(
                go.Scatter(x=self.projections[where, 0], y=self.projections[where, 1],
                        name=str(label), mode='markers'
                ), row=1, col=1)
        self.fig.update_layout(clickmode='event+select')
        # maintain zoom
        if relayout_data is not None:
            if 'xaxis.range[0]' in relayout_data:
                self.fig['layout']['xaxis']['range'] = [
                    relayout_data['xaxis.range[0]'],
                    relayout_data['xaxis.range[1]']
                ]
            if 'yaxis.range[0]' in relayout_data:
                self.fig['layout']['yaxis']['range'] = [
                    relayout_data['yaxis.range[0]'],
                    relayout_data['yaxis.range[1]']
                ]
        # place legend
        self.fig.update_layout(legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01
        ))
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
        self.callback(
            Output('figure', 'figure'),
            Input('switch_displayed_data', 'value'),
            State('figure', 'relayoutData')) (self.state.update_figure)

    def _setup_page(self):
        self.layout = html.Div([
            # PLOTS
            dcc.Graph(
                id='figure',
                figure=self.state.create_fig()
            ),
            dcc.Checklist(
                id='switch_displayed_data',
                options=[
                    {'label': 'Show Predictions', 'value': 'PRED'},
                ],
                value=[]
            ),
            html.Div([
                dcc.Markdown("""
                    **Selected Data**
                """),
                html.Pre(id='data-info')
            ])
        ])

if __name__ == "__main__":
    data = Adult('', False, 500)
    app = Visualization(data)
    app.run_server(debug=False)