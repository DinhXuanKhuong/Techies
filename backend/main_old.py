
import os
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from supabase import create_client, Client
from auth import get_current_user, supabase
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatSession(BaseModel):
    session_id: str
    title: str
    updated_at: str
    message_count: int


class UpdateSessionRequest(BaseModel):
    title: str


@app.get("/test-auth")
async def test_auth(current_user=Depends(get_current_user)):
    return {
        "message": "Authentication successful",
        "user_id": current_user.get("sub"),
        "email": current_user.get("email"),
        "full_payload": current_user
    }


def get_conversation_history(user_id: str, session_id: str, limit: int = 10):
    """Lấy lịch sử chat gần nhất để làm context"""
    try:
        res = supabase.table("chat_history") \
            .select("role, content") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("created_at", desc=False) \
            .limit(limit * 2) \
            .execute()  # limit * 2 vì mỗi turn có 2 messages (user + assistant)

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
    """Tạo title cho session dựa trên tin nhắn đầu tiên"""
    # Rút gọn tin nhắn đầu tiên làm title
    title = first_message.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    return title if title else "Cuộc trò chuyện mới"


def update_or_create_session(user_id: str, session_id: str, first_message: str = None):
    """Cập nhật hoặc tạo session mới"""
    try:
        # Kiểm tra session đã tồn tại chưa
        existing = supabase.table("chat_sessions") \
            .select("session_id") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .execute()

        if existing.data:
            # Cập nhật updated_at
            supabase.table("chat_sessions") \
                .update({"updated_at": datetime.now().isoformat()}) \
                .eq("user_id", user_id) \
                .eq("session_id", session_id) \
                .execute()
        else:
            # Tạo session mới
            title = generate_session_title(first_message) if first_message else "Cuộc trò chuyện mới"
            supabase.table("chat_sessions").insert({
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        print(f"Error updating session: {e}")




@app.post("/chat")
async def chat(
        background_tasks: BackgroundTasks,
        current_user=Depends(get_current_user),
        q: str = Form(...),
        session_id: str = Form(...),
        file: UploadFile = File(None)
):
    user_id = current_user["sub"]

    # llm = ChatOpenAI(
    #     temperature=0.7,
    #     api_key=os.getenv("OPENAI_KEY"),
    #     model="deepseek/deepseek-chat-v3.1:free",
    #     base_url="https://openrouter.ai/api/v1"
    # )
    llm = ChatOpenAI(
        temperature=0.7,
        api_key=os.getenv("OPENAI_KEY"),
        model="openrouter/sonoma-dusk-alpha",
        base_url="https://openrouter.ai/api/v1"
    )

    # Upload ảnh vào Supabase
    image_url = None


    image_url = None
    if file:
        try:
            file_bytes = await file.read()
            file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
            file_path = f"chat_uploads/{uuid.uuid4()}.{file_ext}"

            print(f"Uploading file to: {file_path}")

            # Upload theo docs
            upload_response = supabase.storage.from_("chat-files").upload(
                file=file_bytes,
                path=file_path,
                file_options={
                    "cache-control": "3600",
                    "upsert": "false"
                }
            )

            print(f"Upload response: {upload_response}")

            # Response
            if upload_response and hasattr(upload_response, 'path'):
                # Get public URL
                public_url_response = supabase.storage.from_("chat-files").get_public_url(file_path)
                print(f"Public URL response: {public_url_response}")

                # Lưu Public URL
                if isinstance(public_url_response, dict):
                    image_url = (
                            public_url_response.get("publicURL") or
                            public_url_response.get("publicUrl")
                    )
                else:
                    # Lấy public url
                    image_url = str(public_url_response)

                print(f"Final image_url: {image_url}")
            else:
                print(f"Upload failed or unexpected response: {upload_response}")

        except Exception as e:
            print(f"Error uploading file: {e}")
            image_url = None
    # Lấy lịch sử conversation để có context
    history_messages = get_conversation_history(user_id, session_id)

    # Kiểm tra xem có phải tin nhắn đầu tiên của session không
    is_first_message = len(history_messages) == 0

    # Tạo message chain
    messages = []

    # System message (optional)
    system_msg = SystemMessage(content="Bạn là một AI assistant hữu ích, trả lời bằng tiếng Việt.")
    messages.append(system_msg)

    # Thêm lịch sử chat
    messages.extend(history_messages)

    # Thêm câu hỏi hiện tại

    if image_url:
        # Format message
        content_parts = []

        if q.strip():
            content_parts.append({
                "type": "text",
                "text": q
            })
        else:
            content_parts.append({
                "type": "text",
                "text": "Hãy mô tả chi tiết ảnh này"
            })

        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_url}
        })

        messages.append(HumanMessage(content=content_parts))
    else:
        messages.append(HumanMessage(content=q))

    accumulated = []

    async def generate():
        try:
            # Stream với full context
            async for chunk in llm.astream(messages):
                content = chunk.content or ""
                accumulated.append(content)
                yield content
        finally:
            background_tasks.add_task(save_to_db)

    def save_to_db():
        try:
            supabase.table("chat_history").insert([
                {"user_id": user_id, "session_id": session_id, "role": "user", "content": q, "image_url": image_url},
                {"user_id": user_id, "session_id": session_id, "role": "assistant", "content": "".join(accumulated)},
            ]).execute()

            # Cập nhật hoặc tạo session
            update_or_create_session(user_id, session_id, q if is_first_message else None)
        except Exception as e:
            print(f"Error saving to DB: {e}")

    return StreamingResponse(generate(), media_type="text/plain")
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



@app.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(current_user=Depends(get_current_user)):
    """Lấy danh sách tất cả chat sessions của user"""
    user_id = current_user["sub"]

    try:
        # Lấy danh sách sessions với số lượng tin nhắn
        sessions_res = supabase.table("chat_sessions") \
            .select("session_id, title, updated_at") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()

        sessions = []
        for session in sessions_res.data:
            # Đếm số tin nhắn trong mỗi session
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
    """Tạo session chat mới"""
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


# @app.put("/sessions/{session_id}")
# async def update_session_title(
#         session_id: str,
#         request: UpdateSessionRequest,
#         current_user=Depends(get_current_user)
# ):
#     """Cập nhật title của session"""
#     user_id = current_user["sub"]
#
#     try:
#         supabase.table("chat_sessions") \
#             .update({"title": request.title, "updated_at": datetime.now().isoformat()}) \
#             .eq("user_id", user_id) \
#             .eq("session_id", session_id) \
#             .execute()
#
#         return {"message": "Session title updated successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.delete("/sessions/{session_id}")
# async def delete_session(session_id: str, current_user=Depends(get_current_user)):
#     """Xóa một chat session và tất cả tin nhắn liên quan"""
#     user_id = current_user["sub"]
#
#     try:
#         # Xóa tất cả tin nhắn trong session
#         supabase.table("chat_history") \
#             .delete() \
#             .eq("user_id", user_id) \
#             .eq("session_id", session_id) \
#             .execute()
#
#         # Xóa session
#         supabase.table("chat_sessions") \
#             .delete() \
#             .eq("user_id", user_id) \
#             .eq("session_id", session_id) \
#             .execute()
#
#         return {"message": "Session deleted successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
