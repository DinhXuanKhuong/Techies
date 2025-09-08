import uuid

import streamlit as st
import os

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

supabase = st.session_state.get("supabase")

@st.cache_resource
def init_llm():
    return ChatOpenAI(temperature=0.7, api_key=os.getenv("OPENAI_KEY"), model="deepseek/deepseek-chat-v3.1:free", base_url="https://openrouter.ai/api/v1")
llm = init_llm()



st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Cuộc trò chuyện của bạn")


def get_user_sessions(user_id):
    try:
        response = supabase.table("chat_history").select("session_id").eq("user_id", user_id).execute()
        unique_sessions = {item['session_id'] for item in response.data}
        return list(unique_sessions)
    except Exception as e:
        st.error(f"Error:", e)
        return []


def fetch_session_history(session_id):
    try:
        response = supabase.table("chat_history").select("role", "content").eq("session_id", session_id).order("created_at").execute()
        return response.data
    except Exception as e:
        st.error(f"Error: {e}")
        return []


def save_message(user_id, session_id, role, content):
    try:
        response = supabase.table("chat_history").insert({
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        st.error(f"Error: {e}")
        return []


if 'user_session' not in st.session_state or st.session_state.user_session is None:
    st.warning("Bạn cần đăng nhập để truy cập trang này.")
    st.info("Vui lòng quay lại trang chủ để đăng nhập.")
    st.stop() # Stop execute the rest of the page

user_id = st.session_state.user_session.user.id



# Sidebar
with st.sidebar:
    st.header("Các đoạn chat của bạn")
    if st.button("Bắt đầu hội thoại mới", type="primary"):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    sessions = get_user_sessions(user_id)
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None

    for session in sessions:
        session_label = f"Chat {session[:13]}.."
        if st.button(session_label, key=session):
            st.session_state.current_session_id = session
            st.session_state.messages = fetch_session_history(session)
            st.rerun()
# If not choosing from previous sessions, require creating a new one
if not st.session_state.current_session_id:
    st.info("Hãy bắt đầu cuộc trò chuyện mới từ thanh bên trái.")
else:
    # Present current message session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Processing input
    if prompt := st.chat_input("Ask an question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        save_message(user_id, st.session_state.current_session_id, "user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_langchain_format = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages]
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}")
                ])
                chain = prompt_template | llm
                response = chain.invoke({"input": prompt, "history": history_langchain_format})
                response_content = response.content
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                save_message(user_id, st.session_state.current_session_id, "assistant", response_content)



            