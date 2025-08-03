#!/bin/bash
pip install --no-cache-dir -r requirements.txt
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app