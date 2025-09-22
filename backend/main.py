import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from auth import get_current_user, supabase
# ✅ Gọi graph từ file derm_graph.py
from derm_agent import run_derm_graph

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MODELS ======================

class ChatSession(BaseModel):
    session_id: str
    title: str
    updated_at: str
    message_count: int


class UpdateSessionRequest(BaseModel):
    title: str


# ================== AUTH ========================

@app.get("/test-auth")
async def test_auth(current_user=Depends(get_current_user)):
    return {
        "message": "Authentication successful",
        "user_id": current_user.get("sub"),
        "email": current_user.get("email"),
        "full_payload": current_user
    }

# ================== HELPERS ======================

def get_conversation_history(user_id: str, session_id: str, limit: int = 10):
    """Lấy lịch sử chat gần nhất để làm context"""
    try:
        res = supabase.table("chat_history") \
            .select("role, content") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .limit(limit * 2) \
            .execute()

        messages = []
        for msg in res.data:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        return messages
    except Exception as e:
        print(f"Error loading history: {e}")
        return []


def generate_session_title(first_message: str) -> str:
    title = first_message.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    return title if title else "Cuộc trò chuyện mới"


# def update_or_create_session(user_id: str, session_id: str, first_message: str = None):
#     """Cập nhật hoặc tạo session mới"""
#     try:
#         existing = supabase.table("chat_sessions") \
#             .select("session_id") \
#             .eq("user_id", user_id) \
#             .eq("session_id", session_id) \
#             .execute()
#
#         if existing.data:
#             supabase.table("chat_sessions") \
#                 .update({"updated_at": datetime.now().isoformat()}) \
#                 .eq("user_id", user_id) \
#                 .eq("session_id", session_id) \
#                 .execute()
#         else:
#             title = generate_session_title(first_message) if first_message else "Cuộc trò chuyện mới"
#             supabase.table("chat_sessions").insert({
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "title": title,
#                 "created_at": datetime.now().isoformat(),
#                 "updated_at": datetime.now().isoformat()
#             }).execute()
#     except Exception as e:
#         print(f"Error updating session: {e}")
def update_or_create_session(user_id: str, session_id: str, first_message: str = None):
    """
    Cập nhật session hiện có hoặc tạo session mới.
    Logic được nâng cấp để cập nhật title nếu nó là title mặc định.
    """
    try:
        # Lấy thông tin session hiện có, bao gồm cả title
        existing_res = supabase.table("chat_sessions") \
            .select("session_id, title") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .execute()

        if existing_res.data:
            # Session đã tồn tại
            session_data = existing_res.data[0]
            update_payload = {"updated_at": datetime.now().isoformat()}

            # MỚI: Chỉ cập nhật title nếu nó là title mặc định và có tin nhắn mới
            if session_data.get("title") == "Cuộc trò chuyện mới" and first_message:
                update_payload["title"] = generate_session_title(first_message)

            supabase.table("chat_sessions") \
                .update(update_payload) \
                .eq("user_id", user_id) \
                .eq("session_id", session_id) \
                .execute()
        else:
            # Session chưa tồn tại, tạo mới như bình thường
            title = generate_session_title(first_message) if first_message else "Cuộc trò chuyện mới"
            supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        print(f"Error updating or creating session: {e}")
# ================== CHAT ==========================

@app.post("/chat")
async def chat(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    q: str = Form(...),
    session_id: str = Form(...),
    file: UploadFile = File(None)
):
    user_id = current_user["sub"]
    image_url = None

    # ===== Upload ảnh vào Supabase Storage =====
    if file:
        try:
            file_bytes = await file.read()
            file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            file_path = f"chat_uploads/{uuid.uuid4()}.{file_ext}"
            print("Uploaded image URL:", image_url)

            res = supabase.storage.from_("chat-files").upload(
                file_path,
                file_bytes,
                {"cache-control": "3600", "upsert": "false"}
            )
            print("Upload response:", res)

            public = supabase.storage.from_("chat-files").get_public_url(file_path)
            image_url = public
            print("Uploaded image URL:", image_url)

        except Exception as e:
            print(f"Error uploading file: {e}")
            image_url = None

    # # ===== Gọi graph xử lý thay vì gọi ChatOpenAI =====
    # accumulated = []
    #
    #
    # async def generate():
    #     try:
    #         result = await run_derm_graph(q, image_url)  # gọi graph
    #         accumulated.append(result)
    #         yield result
    #     finally:
    #         background_tasks.add_task(save_to_db)
    #
    # def save_to_db():
    #     try:
    #         supabase.table("chat_history").insert([
    #             {"user_id": user_id, "session_id": session_id, "role": "user", "content": q, "image_url": image_url},
    #             {"user_id": user_id, "session_id": session_id, "role": "assistant", "content": "".join(accumulated)},
    #         ]).execute()
    #         update_or_create_session(user_id, session_id, q)
    #     except Exception as e:
    #         print(f"Error saving to DB: {e}")
    #
    # return StreamingResponse(generate(), media_type="text/plain")

    # ===== GỌI GRAPH VÀ TRUYỀN LỊCH SỬ CHAT =====
    accumulated_answer = []
    final_rag_docs = []  # Biến để lưu context

    async def generate():
        nonlocal final_rag_docs
        try:
            # BƯỚC 1: Lấy lịch sử chat từ DB
            history = get_conversation_history(user_id, session_id)

            # BƯỚC 2: Gọi graph với câu hỏi mới và lịch sử cũ
            result_dict = await run_derm_graph(q, image_url, chat_history=history)

            answer = result_dict.get("answer", "Xin lỗi, đã có lỗi xảy ra.")
            final_rag_docs = result_dict.get("rag_docs", [])  # Lưu lại rag_docs

            accumulated_answer.append(answer)
            yield answer
        finally:
            # BƯỚC 3: Lưu lại câu hỏi mới và câu trả lời mới vào DB
            background_tasks.add_task(save_to_db)

    def save_to_db():
        try:
            # Chỉ lưu câu trả lời cuối cùng, không phải toàn bộ stream
            final_answer = "".join(accumulated_answer)
            supabase.table("chat_history").insert([
                {"user_id": user_id, "session_id": session_id, "role": "user", "content": q,
                 "image_url": image_url},
                {"user_id": user_id, "session_id": session_id, "role": "assistant", "content": final_answer},
            ]).execute()
            update_or_create_session(user_id, session_id, q)
        except Exception as e:
            print(f"Error saving to DB: {e}")

    return StreamingResponse(generate(), media_type="text/plain")


# ================== HISTORY ===========================

@app.get("/history")
async def get_history(session_id: str, current_user=Depends(get_current_user)):
    user_id = current_user["sub"]

    try:
        res = supabase.table("chat_history") \
            .select("role, content, image_url, created_at") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .execute()
        return res.data
    except Exception as e:
        return {"error": str(e)}

# ================== SESSIONS ==========================

@app.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(current_user=Depends(get_current_user)):
    user_id = current_user["sub"]

    try:
        sessions_res = supabase.table("chat_sessions") \
            .select("session_id, title, updated_at") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()

        sessions = []
        for session in sessions_res.data:
            count_res = supabase.table("chat_history") \
                .select("id", count="exact") \
                .eq("user_id", user_id) \
                .eq("session_id", session["session_id"]) \
                .execute()

            sessions.append({
                "session_id": session["session_id"],
                "title": session["title"],
                "updated_at": session["updated_at"],
                "message_count": count_res.count or 0
            })

        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions")
async def create_new_session(current_user=Depends(get_current_user)):
    user_id = current_user["sub"]
    new_session_id = str(uuid.uuid4())

    try:
        supabase.table("chat_sessions").insert({
            "user_id": user_id,
            "session_id": new_session_id,
            "title": "Cuộc trò chuyện mới",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).execute()

        return {"session_id": new_session_id, "title": "Cuộc trò chuyện mới"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
