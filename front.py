# -*- coding: utf-8 -*-
"""
@author: david restrepo rivera
"""
import streamlit as st
from app.pipeline import EmotionPerceptionTool, FER_LABELS
from app.backend import VideoSource
import numpy as np
from ploting import generate_barplot, generate_lineplot

perception = EmotionPerceptionTool(model_path='./app/IOEPT_v0.1.pt')
camera = VideoSource("/media/lecun/HD/HuMath/Video_reconocimiento de emociones.mp4")

st.title("HuMath Emotion Perception Tool")
run = st.checkbox('Start cam streaming for live emotion detection')
FRAME_WINDOW = st.image([])
caption = st.caption("Start cam streaming to get predictions", unsafe_allow_html=False)

col1, col2 = st.columns([4, 6])
data = np.random.randn(10, 1)

chart = generate_barplot()
barplot = col1.plotly_chart(chart, use_container_width=True)

line = generate_lineplot()
lineplot = col2.plotly_chart(line, use_container_width=True)

def start_streaming_loop():
    session = []
    while run:
        frame = camera.read()
        response = perception(frame)
        FRAME_WINDOW.image(response['image'])
        ### TODO: try-cath KeyError ['data']

        try:
            probs = response['data']['probs'][0]
            caption_text = f"{response['data']['emotions']} detected with {probs.max()}% probability"

            caption.text(caption_text)
            session.append(probs)

            chart = generate_barplot(probs)
            barplot.plotly_chart(chart, use_container_width=True)

            line = generate_lineplot(session)
            lineplot.plotly_chart(line, use_container_width=True)
        except KeyError:
            pass

if run:
    camera.start()
    start_streaming_loop()
    camera.stop()