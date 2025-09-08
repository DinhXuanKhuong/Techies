# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# --- INIT ---
load_dotenv()
st.set_page_config(page_title="Chatbot App", layout="wide")

@st.cache_resource
def init_db():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

if 'supabase' not in st.session_state:
    st.session_state.supabase = init_db()

if 'user_session' not in st.session_state:
    st.session_state.user_session = None

# --- NAVIGATION ---

pages = [
    st.Page("pages/home.py", title="Trang chủ", icon="🏠"),
    st.Page("pages/chat.py", title="Chat", icon="🤖")
]

# if st.session_state.user_session:
#     pages.append(st.Page("pages/chat.py", title="Chat", icon="🤖"))

pg = st.navigation(pages)

pg.run()