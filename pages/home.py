# views/home.py
import streamlit as st

# Get database instance
supabase = st.session_state.get("supabase")

# --- CONFIG ---
st.set_page_config(page_title="DermAI - Hệ thống chẩn đoán da liễu", page_icon="🩺", layout="centered")

# --- HERO SECTION ---
st.markdown(
    """
    <div style="text-align:center; padding:20px 0;">
        <h1>🩺 DermAI</h1>
        <h3>Trợ lý AI hỗ trợ chẩn đoán và chăm sóc da liễu</h3>
        <p>
            Sử dụng trí tuệ nhân tạo để phân tích hình ảnh tổn thương da, gợi ý chẩn đoán sơ bộ 
            và kết nối đến cơ sở y tế gần bạn. 
            <br>
            <b>Giúp cộng đồng tiếp cận chăm sóc y tế nhanh chóng, chính xác và tiện lợi.</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()

# --- GIAO DIỆN TRANG CHỦ ---


if st.session_state.user_session is None:
    col1, col2 = st.columns(2)

    st.info("Vui lòng đăng nhập hoặc đăng ký để tiếp tục.")

    tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "🆕 Đăng ký"])

    # Login
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mật khẩu", type="password", key="login_password")
            submitted = st.form_submit_button("Đăng nhập")

            if submitted:
                try:
                    with st.spinner("Đang đăng nhập..."):
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.user_session = res
                        st.switch_page("pages/chat.py")
                except Exception as e:
                    st.error("Đăng nhập thất bại. Vui lòng kiểm tra email/mật khẩu.")
    # Register
    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Mật khẩu", type="password", key="register_password")
            submitted = st.form_submit_button("Đăng kí")

            if submitted:
                try:
                    supabase.auth.sign_up({"email": email, "password": password})
                    st.success(
                        "Đăng ký thành công! Vui lòng kiểm tra email để xác thực và sau đó đăng nhập."
                    )
                except Exception as e:
                    st.error("Không thể đăng ký. Vui lòng thử lại.")
else:
    user_email = st.session_state.user_session.user.email
    st.success(f"Đăng nhập thành công với tài khoản: **{user_email}**")
    st.markdown("---")
    st.info("👈 Vui lòng chọn trang **Chat** từ thanh bên trái để bắt đầu.")

    if st.button("Đăng xuất"):
        supabase.auth.sign_out()
        st.session_state.user_session = None
        # Clear all session data
        for key in ['current_session_id', 'messages']:
            if key in st.session_state:
                del st.session_state[key]

        for key in list(st.session_state.keys()):
            if key not in ["supabase"]:
                del st.session_state[key]
        st.rerun()



