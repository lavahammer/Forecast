import streamlit as st

st.set_page_config(page_title ="Forecast App",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")

tabs = ["Application","About"]
page = st.sidebar.radio("Tabs",tabs)

