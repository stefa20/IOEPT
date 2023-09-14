# -*- coding: utf-8 -*-
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import flask
import base64
import cv2
import os
from time import strftime, gmtime

def get_time_axis(user):

    fps = session['info'].fps[0]
    frame_lenght = len(session['users'][user])
    dx = 1/fps
    total_secs = frame_lenght * dx
    time = strftime('%H:%M:%S', gmtime(total_secs))
    time_range = pd.date_range(start='00:00:00', end=time, periods = frame_lenght)

    return time_range

def create_line_plot(user):
    time_ax = get_time_axis(user)
    tick_axis = time_ax.ceil('s').unique().time
    data = session['users'][user].copy().set_index(time_ax.time)
    lineplt = px.line(data.rolling(window=100).mean(),
                   labels={'index': "Time", 'value': "Expression probability", 'variable': "Emotion"})
    lineplt.update_xaxes(tickvals=tick_axis)
    return lineplt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,  meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])

### Read data
sess_path = './sessions/07-01-21_21:57'
session = {}
sess_info = pd.read_csv(f"{sess_path}/{sess_path.rsplit('/',1)[-1]}_info.csv")
session['info'] = sess_info
session['users'] = {}
for user_result in os.listdir(sess_path):
    if 'user' in user_result:
        data = pd.read_csv(f"{sess_path}/{user_result}")
        user = user_result.rsplit('_', 1)[0]
        session['users'][user] = data
print(session['info'])
session_str = [f'{l}: {v}' for l,v in zip(list(session['info'].columns), session['info'].values[0])]
session_str = f'Session Information: {session_str[0]}, {session_str[1]}'

## Create radio options for users
radio_values = users = list(session['users'].keys())
radio_labels = [f'User {i+1}' for i in range(len(users))]
radio_options = [{'label': label, 'value':value} for label, value in zip(radio_labels, radio_values)]

## Create lineplot for user
user = 'user0'
fig2 = create_line_plot(user)

## Read test image ( replace for video)
test_png = './sample.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


app.layout = html.Div(children=[
    html.H1(children='Reporte de emociones'),

    html.Div(children=session_str),

    # html.Div([
    #     html.Img(src='data:image/png;base64,{}'.format(test_base64)),
    # ]),
    html.Div([
        html.Video(src='./static/processed.mp4', controls=True,
                   style={'width':'100%'}, preload='meta'),
    ]),

    dcc.RadioItems(
        id='radio-users',
        options=radio_options,
        value=user,
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='line-plot',
            figure=fig2
    ))

])

@app.callback(
    Output(component_id='line-plot', component_property='figure'),
    Input(component_id='radio-users', component_property='value')
)
def update_output_div(input_value):
    plot = create_line_plot(input_value)
    return plot


## Need to stream video to html Video class
server = app.server

@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)

if __name__ == '__main__':
    app.run_server(debug=True)