import json

from umap import UMAP
import numpy as np
import torch

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataset.adult import Adult
from model import Net
from tree_ensemble import RandomForest

class VisualState:

    def __init__(self, data, models):
        self.data = data
        self.models = models
        self.current_model = models[0].__class__.__name__
        self.umap_2d = UMAP(random_state=0)
        self.projections = self.umap_2d.fit_transform(self.data.numpy()[0]) # TODO maybe better use learned model embeddings here?!
        self.predict()

    def predict(self):
        for model in self.models:
            if model.__class__.__name__ == self.current_model:
                # TODO replace by predict function call
                self.pred = np.array(torch.argmax(model(self.data.torch()[0]), dim=-1))
                self.confuse = np.zeros_like(self.pred).astype(str)
                correct = self.data.numpy()[1] == self.pred
                self.confuse[correct] = 'CORRECT'
                self.confuse[np.invert(correct)] = 'WRONG'

    def get_point_id(self, interaction_data):
        curve_number = interaction_data['curveNumber']
        label = float(self.fig['data'][curve_number]['name'])
        return np.where(self.data.numpy()[1] == label)[0][interaction_data['pointNumber']]

    def update_figure(self, display_radio, n_clicks, chosen_model, relayout_data, selected_data, table_data):
        changed_id = dash.callback_context.triggered[0]['prop_id']
        if changed_id == 'apply-button.n_clicks':
            self.update_data(selected_data, table_data)
        if changed_id == 'model-select.value':
            self.current_model = chosen_model
            self.predict()
        if 'PRED' in display_radio:
            return self.create_fig(use_prediction=True, relayout_data=relayout_data)
        return self.create_fig(relayout_data=relayout_data)

    def update_data(self, selected_data, table_data):
        if selected_data is not None:
            sel_idcs = []
            for point in selected_data['points']:
                point_id = int(self.get_point_id(point))
                sel_idcs.append(point_id)
        for i,sel in enumerate(sel_idcs):
            del table_data[i]['pred']
            continuous_indices = [0, 2, 4, 10, 11, 12]
            normalized_row = []
            for i,e in enumerate(table_data[i].values()):
                if i in continuous_indices: normalized_row.append(int(e))
                else: normalized_row.append(e)
            normalized_row = self.data.normalize_single(normalized_row)
            print('before', self.data.pd_X.iloc[sel])
            self.data.pd_X.iloc[sel] = normalized_row
            print('after', self.data.pd_X.iloc[sel])

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
        self.fig.update_layout(
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.01
            ),
            margin={'t': 0, 'b': 0, 'l': 0, 'r': 0})
        return self.fig

    def update_selected_data(self, selected_data):
        if selected_data is not None:
            rep = {}
            for point in selected_data['points']:
                point_id = int(self.get_point_id(point))
                rep[point_id] = self.data.as_json(point_id)
            return json.dumps(rep, indent=4)
        return json.dumps(None)

    def update_table(self, selected_data):
        if selected_data is not None:
            sel_idcs = []
            for point in selected_data['points']:
                point_id = int(self.get_point_id(point))
                sel_idcs.append(point_id)
            pdout = self.data.pd_X.iloc[sel_idcs]
            for i in range(len(sel_idcs)):
                denorm = self.data.denormalize(pdout.iloc[i].to_numpy())
                pdout.iloc[i] = denorm
            pdout['pred'] = [self.pred[i] for i in sel_idcs]
            return pdout.to_dict('records')

    def update_shap_fig(self, selected_data=None):
        self.fig_shap = go.Figure(layout=go.Layout(
            margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
        ))
        col_names = self.data.get_column_names()
        for model in self.models:
            if model.__class__.__name__ == self.current_model:
                if selected_data is not None:
                    sel_idcs = []
                    for point in selected_data['points']:
                        point_id = int(self.get_point_id(point))
                        sel_idcs.append(point_id)
                    # TODO shap_values = model.get_shap(sel_idcs)
                    shap_values = np.random.random(len(sel_idcs) * len(col_names)).reshape((len(sel_idcs), len(col_names)))
                    shap_values = np.mean(shap_values, axis=0)
                else: # nothing selected
                    shap_values = [0 for _ in col_names]
                self.fig_shap.add_trace(go.Bar(x=col_names, y=shap_values))
        self.fig_shap.update_xaxes(type='category')
        return self.fig_shap


class Visualization(dash.Dash):

    def __init__(self, data, models):
        super().__init__(__name__)#, external_stylesheets=external_stylesheets)
        self.data = data
        self.state = VisualState(data, models)
        self.model_names = [model.__class__.__name__ for model in models]
        self._setup_page()
        self.callback(
            Output('figure', 'figure'),
            [Input('switch_displayed_data', 'value'), Input('apply-button', 'n_clicks'), Input('model-select', 'value')],
            [State('figure', 'relayoutData'), State('figure', 'selectedData'), State('table', 'data')]) (self.state.update_figure)
        self.callback(
            Output('table', 'data'),
            Input('figure', 'selectedData')) (self.state.update_table)
        self.callback(
            Output('fig-shapley', 'figure'),
            Input('figure', 'selectedData')) (self.state.update_shap_fig)


    def _setup_page(self):
        self.layout = html.Div(className='cont-grid', children=[
            html.Div(className='cont-graph', children=[
                dcc.Graph(
                    id='figure',
                    responsive=True,
                    config={'responsive': True},
                    style={'height': '100%', 'width': '100%'},
                    figure=self.state.create_fig()
                ),
            ]),
            html.Div(className='cont-graph-custom', children=[
                dcc.Checklist(
                    id='switch_displayed_data',
                    options=[
                        {'label': 'Show Predictions', 'value': 'PRED'},
                    ],
                    value=[]
                ),
                dcc.Dropdown(
                    id='model-select',
                    options=[{'label': mname, 'value': mname} for mname in self.model_names],
                    value=self.model_names[0]
                ),      
            ]),
            # Table
            html.Div(className='cont-table', children=[
                dash_table.DataTable(
                    id='table',
                    data=None,
                    columns = [{"name": col, "id": col, 'presentation': 'dropdown'} if (col in self.data.get_categorical_column_names()) else {"name": col, "id": col} for col in self.data.get_column_names()] + [{"name": 'pred', "id": 'pred'}],
                    editable=True,
                    dropdown = {category:{'options': [{'label': i, 'value': i} for i in choices]} for category,choices in self.data.get_categorical_choices().items()},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{pred} = 0', 'column_id': 'pred'},
                            'backgroundColor': 'blue', 'color': 'white'
                        },
                        {
                            'if': {'filter_query': '{pred} = 1', 'column_id': 'pred'},
                            'backgroundColor': 'red', 'color': 'white'
                        },
                    ]
                ),
                html.Div(id='table-dropdown-container')
            ]),
            # Apply button
            html.Div(className='cont-apply', children=[
                html.Button('Apply changes', id='apply-button', n_clicks=0),
            ]),
            html.Div(className='cont-shapley', children=[
                dcc.Graph(
                    id='fig-shapley',
                    responsive=True,
                    config={'responsive': True},
                    style={'height': '100%', 'width': '100%'},
                    figure=self.state.update_shap_fig()
                ),
            ])
        ])

if __name__ == "__main__":
    data = Adult('', False, 500)
    models = [Net(), RandomForest()]
    app = Visualization(data, models)
    app.run_server(debug=False)
