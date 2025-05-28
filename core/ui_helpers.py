# ---------- core/ui_helpers.py ----------
import streamlit as st

def set_mobile_styles():
    st.markdown("""
    <style>
        button {
            min-height: 3em !important;
            padding: 1em !important;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            width: 100% !important;
        }
        .stImage > img {
            max-width: 100% !important;
            height: auto !important;
        }
        @media (max-width: 768px) {
            .hide-mobile {
                display: none;
            }
        }
    </style>
    """, unsafe_allow_html=True)
