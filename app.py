import json

from umap import UMAP
import numpy as np
import torch
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dataset.adult import Adult, column_names
from model import FcNet, ConvNet
from tree_ensemble import RandomForest

class VisualState:

    def __init__(self, data, models):
        self.data = data
        self.models = models
        self.current_model = models[0].__class__.__name__
        self.umap_2d = UMAP(random_state=0)
        self.projections = self.umap_2d.fit_transform(self.data.numpy()[0]) # TODO maybe better use learned model embeddings here?!
        self.predict()
        self.y = self.pred
        self.create_fig()
        self.selection = []
        self.remap_idc = {}

    def predict(self, sel_idcs=None):
        for model in self.models:
            if model.__class__.__name__ == self.current_model:
                if sel_idcs is not None:
                    self.pred[sel_idcs] = model.predict(self.data, sel_idcs)
                else:
                    self.pred = model.predict(self.data, sel_idcs)
                self.confuse = np.zeros_like(self.pred).astype(str)
                correct = self.data.numpy()[1] == self.pred
                self.confuse[correct] = 'CORRECT'
                self.confuse[np.invert(correct)] = 'WRONG'

    def get_point_id(self, interaction_data):
        curve_number = interaction_data['curveNumber']
        label = np.array([self.fig['data'][curve_number]['name']], dtype=self.y.dtype)
        p_id = np.where(self.y == label)[0][interaction_data['pointNumber']]
        return p_id

    def get_points_ids(self, selected_data):
        # TODO add a check, that when any of the edited points (listed in remap_idc)
        #      is deselected, its entry is also removed from remap_idc (it won't be selectable again!)
        sel_idcs = []
        for point in selected_data['points']:
            try:
                point_id = self.remap_idc[(point['curveNumber'], point['pointNumber'])]
            except KeyError:
                point_id = int(self.get_point_id(point))
            sel_idcs.append(point_id)
        return sel_idcs

    def update_figure(self, selected_data, display_radio, n_clicks, chosen_model, relayout_data, table_data):
        changed_id = dash.callback_context.triggered[0]['prop_id']
        if changed_id == 'figure.selectedData':
            self.selection = self.get_points_ids(selected_data)
        if changed_id == 'apply-button.n_clicks':
            self.update_data(table_data)
            # store remapping for updated points
            self.remap_idc = {}
            for idx, point in enumerate(selected_data['points']):
                self.remap_idc[(point['curveNumber'], point['pointNumber'])] = self.selection[idx]
        if changed_id == 'model-select.value':
            self.current_model = chosen_model
            self.predict()
        if 'PRED' in display_radio:
            self.y = self.confuse
        else:
            self.y = self.pred
        if changed_id != 'figure.selectedData':
            self.create_fig(relayout_data=relayout_data)
        # set selection in figure
        for label in np.unique(self.y):
            label_idc = np.where(self.y == label)[0]
            select = np.array([np.where(label_idc == point)[0] for point in self.selection if np.any(label_idc == point)]).flatten()
            self.fig.update_traces(selectedpoints=select, selector=dict(name=str(label)))
        return self.fig, self.selection

    def update_counterfactuals(self, n_clicks):
        if len(self.selection) != 0:
            for model in self.models:
                if model.__class__.__name__ == self.current_model:
                    cfs = model.get_counterfactual(self.selection, self.pred, self.data)
                    prediction = pd.DataFrame(data=cfs['income'])
                    prediction.columns = ['pred']
                    cfs = cfs.drop(['income', 'index'], axis=1)
                    cfs = self.data.denormalize(cfs)
                    return pd.concat((cfs, prediction), axis=1).to_dict('records')

    def update_data(self, table_data):
        if len(self.selection) != 0:
            for i, sel in enumerate(self.selection):
                del table_data[i]['pred']
                continuous_indices = [0, 2, 4, 10, 11, 12]
                normalized_row = []
                for i,e in enumerate(table_data[i].values()):
                    if i in continuous_indices: normalized_row.append(int(e))
                    else: normalized_row.append(e)
                normalized_row = self.data.normalize_single(normalized_row)
                self.data.pd_X.iloc[sel] = normalized_row
                self.projections[sel] = self.umap_2d.transform(self.data.numpy()[0][sel].reshape(1, -1))
            self.predict(self.selection)

    def create_fig(self, relayout_data=None):
        self.fig = go.Figure(layout=go.Layout(margin={'t': 0, 'b': 0, 'l': 0, 'r': 0}))
        # plot traces
        for label in np.unique(self.y):
            label_idc = np.where(self.y == label)[0]
            self.fig.add_trace(
                go.Scatter(x=self.projections[label_idc, 0], y=self.projections[label_idc, 1],
                           name=str(label), mode='markers'
                ))
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
        # set legend position and selectability
        self.fig.update_layout(
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.01
            ),
            clickmode='event+select'
        )

    def update_table(self, resetbutton_nclick, tmp=None):
        if len(self.selection) != 0:
            pdout = self.data.pd_X.iloc[self.selection]
            for i in range(len(self.selection)):
                denorm = self.data.denormalize(pdout.iloc[i].to_numpy())
                pdout.iloc[i] = denorm
            pdout['pred'] = [self.pred[i] for i in self.selection]
            return pdout.to_dict('records')

    def update_shap_fig(self, tmp=None):
        self.fig_shap = go.Figure(layout=go.Layout(margin={'t': 0, 'b': 0, 'l': 0, 'r': 0}))
        col_names = self.data.get_column_names()
        for model in self.models:
            if model.__class__.__name__ == self.current_model:
                if len(self.selection) != 0:
                    shap_values = model.get_shap(self.selection, self.pred, self.data)
                    shap_mean = np.mean(shap_values, axis=0)
                    shap_error = np.std(shap_values, axis=0)
                else: # nothing selected
                    shap_mean = [0 for _ in col_names]
                    shap_error = [0 for _ in col_names]
                self.fig_shap.add_trace(go.Bar(x=col_names, y=shap_mean, error_y=dict(type='data', array=shap_error)))
        self.fig_shap.update_xaxes(type='category')
        return self.fig_shap


class Visualization(dash.Dash):

    def __init__(self, data, models):
        super().__init__(__name__)
        self.data = data
        self.state = VisualState(data, models)
        self.model_names = [model.__class__.__name__ for model in models]
        self._setup_page()
        self.callback(
            [Output('figure', 'figure'), Output('tmp', 'children')],
            [Input('figure', 'selectedData'), Input('switch_displayed_data', 'value'),
             Input('apply-button', 'n_clicks'), Input('model-select', 'value')],
            [State('figure', 'relayoutData'), State('table', 'data')]) (self.state.update_figure)
        self.callback(
            Output('table', 'data'),
            [Input('reset-button', 'n_clicks'), Input('tmp', 'children')]) (self.state.update_table)
        self.callback(
            Output('fig-shapley', 'figure'),
            [Input('tmp', 'children')]) (self.state.update_shap_fig)
        self.callback(
            Output('counterfactual-table', 'data'),
            [Input('generate-counterfactuals', 'n_clicks')]) (self.state.update_counterfactuals)


    def _setup_page(self):
        self.layout = html.Div(className='cont-grid', children=[
            html.Div(className='cont-graph', children=[
                dcc.Graph(
                    id='figure',
                    responsive=True,
                    config={'responsive': True},
                    style={'height': '100%', 'width': '100%'},
                    figure=self.state.fig
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
            html.Br(),
            html.P("Selected data:"),
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
                html.P("Counterfactuals:"),
                dash_table.DataTable(
                    id='counterfactual-table',
                    data=None,
                    columns = [{"name": col, "id": col, 'presentation': 'dropdown'} if (col in self.data.get_categorical_column_names()) else {"name": col, "id": col} for col in self.data.get_column_names()] + [{"name": 'pred', "id": 'pred'}],
                    editable=False,
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
                html.Button('Reset', id='reset-button', n_clicks=0),
                html.Button('Apply changes', id='apply-button', n_clicks=0),
                html.Button('Generate Counterfactuals', id='generate-counterfactuals', n_clicks=0),
            ]),
            html.Div(className='cont-shapley', children=[
                dcc.Graph(
                    id='fig-shapley',
                    responsive=True,
                    config={'responsive': True},
                    style={'height': '100%', 'width': '100%'},
                    figure=self.state.update_shap_fig()
                ),
                html.Div(id='tmp', style={'display': 'none'})
            ])
        ])

if __name__ == "__main__":
    data = Adult('', False, 500)
    models = [RandomForest(), FcNet()]
    app = Visualization(data, models)
    app.run_server(debug=False)
