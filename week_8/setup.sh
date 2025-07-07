#!/bin/bash

mkdir -p ~/.streamlit
echo "[general]"  > ~/.streamlit/config.toml
echo "email = \"\""  >> ~/.streamlit/config.toml
echo "[server]"  > ~/.streamlit/config.toml
echo "headless = true"  >> ~/.streamlit/config.toml
echo "port = $PORT"  >> ~/.streamlit/config.toml
echo "enableCORS = false"  >> ~/.streamlit/config.toml 