"""
Robotron with ML — Streamlit launcher
Embeds the game (game/index.html) in a Streamlit page.
Run: streamlit run app.py
"""

import json
import os
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Robotron with ML",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Path to the game HTML (same folder as this app's parent)
GAME_HTML = Path(__file__).resolve().parent / "game" / "index.html"
MARL_POLICY_JSON = Path(__file__).resolve().parent / "game" / "marl_policy.json"
DNN_PREDICTOR_JSON = Path(__file__).resolve().parent / "game" / "dnn_predictor.json"

if not GAME_HTML.exists():
    st.error(f"Game file not found: {GAME_HTML}")
    st.stop()

with open(GAME_HTML, "r", encoding="utf-8") as f:
    game_html = f.read()

inject_scripts = []
if MARL_POLICY_JSON.exists():
    with open(MARL_POLICY_JSON, "r", encoding="utf-8") as f:
        inject_scripts.append("window.MARL_POLICY=" + json.dumps(json.load(f)))
if DNN_PREDICTOR_JSON.exists():
    with open(DNN_PREDICTOR_JSON, "r", encoding="utf-8") as f:
        inject_scripts.append("window.DNN_PREDICT=" + json.dumps(json.load(f)))
# Docker: show image (or image ID) and container id at bottom of canvas
_docker_env = Path("/.dockerenv").exists() or os.environ.get("container")
if _docker_env:
    _img_id = os.environ.get("DOCKER_IMAGE_ID")
    _img_name = os.environ.get("DOCKER_IMAGE_NAME", "robotron-ml")
    _img_part = ("Image ID: " + _img_id[:20] + "…") if _img_id else ("Image: " + _img_name)
    _cid = (os.environ.get("HOSTNAME") or "")[:12]
    _docker_label = "Launched from Docker · " + _img_part + (" · Container: " + _cid if _cid else "")
    inject_scripts.append("window.DOCKER_LABEL=" + json.dumps(_docker_label))
if inject_scripts:
    game_html = game_html.replace(
        "<body>",
        "<body>\n<script>" + ";".join(inject_scripts) + ";</script>",
        1,
    )

st.markdown(
    '<div style="text-align: center;">'
    '<h3 style="margin-bottom: 0.25rem;">Robotron with ML</h3>'
    '<p style="margin: 0.25rem 0; font-size: 0.9rem;">'
    '<strong>Controls</strong> (click the game area first so keys work): '
    '↑↓←→ Move · Space Fire · S Smart missile · F Shield (hold) · G New walls</p>'
    '<p style="margin: 0.25rem 0; font-size: 0.85rem; color: #8f8;">'
    '<strong>NeuroGamingLab</strong> · Designer & architect by Tuệ Hoàng, AI/ML Engineer</p>'
    '<p style="margin: 0.25rem 0; font-size: 0.8rem; color: #6a6;">Multi-LLM–assisted development.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Embed the full game (900x600 canvas + UI). Wide layout so full canvas is visible.
st.components.v1.html(game_html, height=650, scrolling=False)
