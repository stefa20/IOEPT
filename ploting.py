import pandas as pd
import plotly.express as px
import numpy as np
from app.pipeline import FER_LABELS

def generate_barplot(heights= np.zeros(len(FER_LABELS))):
    frame = pd.DataFrame({'labels': FER_LABELS, 'probs': heights})
    chart = px.bar(frame, x='probs', y='labels', orientation='h')
    chart.update_yaxes({"side": 'right', 'showgrid': False, 'title':None})
    chart.update_xaxes({'autorange':'reversed', 'showgrid': False, 'title': 'Probs'})
    return chart

def generate_lineplot(probs= np.zeros([1, len(FER_LABELS)])):
    frame = pd.DataFrame(probs, columns=FER_LABELS)
    chart = px.line(data_frame=frame)
    chart.update_yaxes({'showgrid': False, 'title':None})
    chart.update_xaxes({'showgrid': False, 'title': 'Frame'})
    return chart