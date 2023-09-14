#!/bin/bash

uvicorn server:app --host 0.0.0.0 --port 80 &
streamlit run front.py --server.port 82

