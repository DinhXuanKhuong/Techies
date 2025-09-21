
import os
from dotenv import load_dotenv
from typing import TypedDict, Dict, List, Optional




from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomaly import test_image, Autoencoder
from cv_tool import predict_image, load_model
from rag import retrieve_relevant_chunks, load_titles_from_chroma, vector_store, titles, retriever

# ====== Load ENV ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# ====== Init LLM ======
# llm = ChatOpenAI(
#     temperature=0,
#     api_key=OPENAI_API_KEY,
#     model="meta-llama/llama-4-maverick:free",
#     base_url="https://openrouter.ai/api/v1",
#     max_tokens=3072
# )

llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),   # API key của Groq
    model="meta-llama/llama-4-maverick-17b-128e-instruct",  # hoặc model khác từ Groq
    base_url="https://api.groq.com/openai/v1",
    max_tokens=3072
)

# ====== Init models ======
# cv_model, cv_device = load_model('best_model_final.pth')
# anomaly_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# anomaly_model = torch.load(
#     'autoencoder_skin.pth',
#     map_location=anomaly_device,
#     weights_only=False
# )
# anomaly_model.to(anomaly_device)
# anomaly_model.eval()
#
# criterion = nn.MSELoss()
# threshold = 0.25
# transform = A.Compose([
#     A.Resize(height=224, width=224),
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
#     ToTensorV2()
# ])

# ====== Init models ======
cv_model, cv_device = load_model('best_model_final.pth')

# Global variables để cache model
_anomaly_model = None
_anomaly_device = None
_criterion = None
_threshold = 0.25
_transform = None


def get_anomaly_model():
    global _anomaly_model, _anomaly_device, _criterion, _transform
    if _anomaly_model is None:
        _anomaly_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _anomaly_model = torch.load(
            'autoencoder_skin.pth',
            map_location=_anomaly_device,
            weights_only=False
        )
        _anomaly_model.to(_anomaly_device)
        _anomaly_model.eval()

        _criterion = nn.MSELoss()
        _transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return _anomaly_model, _anomaly_device, _criterion, _transform

# ====== Tools ======
# @tool
# def anomaly_tool(image_path: str) -> dict:
#     """Kiểm tra ảnh có phải tổn thương da không."""
#     error, is_anomaly = test_image(
#         anomaly_model, image_path, transform,
#         criterion, anomaly_device, threshold
#     )
#     return {"reconstruction_error": error, "is_anomaly": is_anomaly}

@tool
def anomaly_tool(image_path: str) -> dict:
    """Kiểm tra ảnh có phải tổn thương da không."""
    anomaly_model, anomaly_device, criterion, transform = get_anomaly_model()
    error, is_anomaly = test_image(
        anomaly_model, image_path, transform,
        criterion, anomaly_device, _threshold
    )
    return {"reconstruction_error": error, "is_anomaly": is_anomaly}

@tool
def cv_tool(image_path: str) -> dict:
    """Dự đoán bệnh từ ảnh da."""
    return predict_image(cv_model, image_path, cv_device)

# @tool
# def knowledge_retrieval_tool(query: str) -> List[str]:
#     """Truy xuất thông tin y tế từ KB."""
#     return retrieve_relevant_chunks(vector_store, titles, query, num_results=5)


# Sửa lại tool để nó đơn giản hơn
@tool
def knowledge_retrieval_tool(query: str) -> List[Document]:
    """Truy xuất thông tin y tế từ KB."""
    # Giờ đây chỉ cần gọi hàm mới
    return retrieve_relevant_chunks(query)

# ...

# ====== State ======
class DermState(TypedDict):
    user_input: Optional[str]
    image: Optional[str]
    cv_results: Optional[Dict]
    symptoms: Optional[List[str]]
    rag_docs: Optional[List[str]]
    final_diagnosis: Optional[str]
    reasoning: Optional[str]
    answer: Optional[str]
    next_action: Optional[str]
    visited_actions: Optional[List[str]]
    anomaly_check: Optional[Dict]
    iterations: Optional[int]
    plan: Optional[List[str]]
    plan_index: Optional[int]
    other_topic: Optional[bool]
    # awaiting_input: Optional[bool]   # nếu agent đang chờ user trả lời

# ====== Structured Models (kept if bạn muốn dùng LLM planner later) ======
class Task(BaseModel):
    name: str = Field(description="Tên step cần chạy")
    description: str = Field(description="Mô tả ngắn gọn")

class Tasks(BaseModel):
    tasks: List[Task] = Field(description="Danh sách subtasks cần chạy theo thứ tự")

class ReasoningResult(BaseModel):
    diagnosis: Optional[str] = Field(description="Tên bệnh phù hợp nhất hoặc None")
    reasoning: str = Field(description="Lý do chọn bệnh")

planner = llm.with_structured_output(Tasks)
reasoner = llm.with_structured_output(ReasoningResult)

# ====== Validate Node ======
def validate_input(state: DermState) -> DermState:
    # state["awaiting_input"] = False
    img = state.get("image")
    if img:
        try:
            result = anomaly_tool.invoke(img)
        except Exception:
            result = {"reconstruction_error": None, "is_anomaly": True}
            print("[Validate] Anomaly tool failed, treating as anomaly.")
        state["anomaly_check"] = result

    # nếu không có ảnh và không có mô tả
    if not img and not (state.get("symptoms") or state.get("user_input")):
        q = llm.invoke(
            "Bạn chưa gửi ảnh và cũng không mô tả gì thêm. "
            "Vui lòng gửi ảnh hoặc mô tả triệu chứng để tôi hỗ trợ."
        )
        state["answer"] = q.content
        # state["awaiting_input"] = True
        state["next_action"] = "end"
    else:
        # Có ảnh hoặc có mô tả -> tiếp tục router để phân loại topic
        state["next_action"] = "router"
    return state


def classify_topic(state: DermState) -> DermState:
    """
    Phân tích toàn bộ input (ảnh + text):
    - Nếu ảnh hợp lệ hoặc text có mô tả triệu chứng -> derm
    - Nếu không -> other
    """
    # Check ảnh
    anomaly = state.get("anomaly_check")
    has_valid_image = bool(state.get("image")) and anomaly is not None and not anomaly.get("is_anomaly", True)

    # Check text có triệu chứng không
    text = (state.get("user_input") or "").strip()
    has_symptom_text = False
    if text:
        # q = llm.invoke(
        #     f"Câu sau đây có mô tả triệu chứng da liễu không (vị trí, thời gian, đau/ ngứa/ rát, lan rộng,...)?\n"
        #     f"'{text}'\n"
        #     "Trả lời chỉ 'yes' hoặc 'no'."
        # )
        q = llm.invoke(
            f"Bạn là một bộ phân loại triệu chứng da liễu.\n"
            f"Nhiệm vụ: cho biết đoạn văn sau có MÔ TẢ TRIỆU CHỨNG DA LIỄU của một người hay không.\n\n"
            f"Định nghĩa (để phân loại): 'mô tả triệu chứng da liễu' = bất kỳ thông tin nào mô tả trực tiếp tình trạng da của một người (hiện tại hoặc đã xảy ra hoặc người liên quan user bị) — bao gồm:\n"
            f" - vị trí cơ thể (ví dụ: tay, chân, mặt, lưng,...),\n"
            f" - dấu hiệu/triệu chứng cụ thể (ngứa, rát, đau, đỏ, nổi mẩn, sưng, bong tróc, chảy dịch, loét, vảy, lan rộng,...),\n"
            f" - thời gian/diễn tiến (mới xuất hiện, kéo dài, nặng hơn, tái phát,...).\n"
            f" - tiền sử của gia đình, người thân (ví dụ: 'Gia đình tôi có tiền sử viêm da.')\n\n"
            f"Quy tắc rõ ràng:\n"
            f" - Trả 'yes' nếu văn bản mô tả trực tiếp các dấu hiệu/triệu chứng hoặc diễn tiến liên quan đến DA (skin) của một người cụ thể (ví dụ: 'Tôi bị ngứa 2 ngày', 'Nổi mẩn ở cổ lan ra sau 1 tuần').\n"
            f" - Trả 'no' nếu văn bản là: những điều về 'xin chào', tâm sự cuộc sống, nói chuyện không về y khoa, hoặc dùng nghĩa bóng/ẩn dụ ('ngứa tay' = muốn làm việc).\n"
            f" - Mô tả triệu chứng trong quá khứ cho chính người nói (ví dụ 'Trước tôi từng bị nổi mẩn') vẫn tính là 'yes'.\n\n"
            f"Ví dụ (ví dụ + đáp án):\n"
            f" - 'Mặt em hôm nay bị đỏ và rát, rất ngứa.'  -> yes\n"
            f" - 'Nếu da tôi ngứa thì phải làm sao?'         -> yes\n"
            f" - 'Xin chào, hello bạn'                      -> no\n"
            f" - 'Tôi buồn quá, ước gì có bạn tâm sự với tôi'     -> no\n"
            f" - 'Tôi cảm thấy áp lực cuộc sống, gia đình đè nén lên tôi, tôi bị đau'     -> no\n"
            f" - 'Tôi bị đau bụng                       '     -> no\n"
            f" - 'Gia đình tôi có tiền sử viêm da.'          -> yes\n\n"
            f"Văn bản: '{text}'\n"
            f"CHỈ trả 'yes' hoặc 'no' (viết thường, không thêm lời giải thích hoặc ký tự khác)."
        )
        ans = q.content.strip().lower()
        print(f"[InputClassifier] Symptom text check answer: {ans}")
        has_symptom_text = "yes" in ans
        state['other_topic'] = not has_symptom_text

    # --- Gộp kết quả ---
    if has_valid_image or has_symptom_text:
        state["next_action"] = "planner"
    else:
        state["next_action"] = "other_topic"

    print(f"[InputClassifier] image_valid={has_valid_image}, symptom_text={has_symptom_text} -> {state['next_action']}")
    return state

# ====== Other Topic Node ======
def other_topic_answer(state: DermState) -> DermState:
    '''
    Nếu là câu hỏi ngoài domain, trả lời thân thiện và yêu cầu user hỏi về da liễu.
    '''
    query = state.get("user_input", "").lower()
    q = llm.invoke(
        f"Người dùng hỏi: {query}\n"
        "Dựa trên câu hỏi này, hãy trả lời một cách tự nhiên, thân thiện và dễ thương rằng bạn là trợ lý ảo về da liễu "
        "Có thể hỏi user về tình trạng da, triệu chứng, hình ảnh da, hoặc các vấn đề chăm sóc da nhé!"
    )
    print(f"[Other Topic] Answer: {q.content}.")
    state["answer"] = q.content
    # state["awaiting_input"] = True
    state["next_action"] = "end"
    return state

# ====== Planner Node ======



def orchestrator_plan(state: DermState) -> DermState:
    """
    Phân loại query của user và xác định plan xử lý dựa trên trạng thái hiện tại.
    """
    plan = []
    img = state.get("image")
    # BỎ DÒNG NÀY: other_topic = state.get("other_topic", False)

    # determine image_valid: anomaly exists and is False
    image_valid = False
    if img:
        anomaly = state.get("anomaly_check")
        if anomaly is None:
            image_valid = True
        else:
            image_valid = not anomaly.get("is_anomaly", True)

    if img and image_valid:
        # hình hợp lệ -> dùng CV rồi hỏi thêm triệu chứng nếu cần
        plan = [
            "cv_inference",
            "llm_ask_symptom",
            "symptom_matching",
            "llm_reasoning",
            "llm_answer"
        ]
    # SỬA LOGIC Ở ĐÂY
    # Nếu không có ảnh hợp lệ, NHƯNG có input text (và đã vượt qua router)
    # thì chạy plan cho text.
    elif state.get("user_input"):
        plan = [
            "symptom_matching",
            "llm_reasoning",
            "llm_answer"
        ]
    else:
        # Các trường hợp còn lại (không có ảnh, không có text) -> plan rỗng
        plan = []

    print(f"[Planner] Generated plan: {plan}")
    state["visited_actions"] = []
    state["plan"] = plan
    state["plan_index"] = 0
    return state

# ====== Executor Node ======
def executor(state: DermState) -> DermState:
    plan = state.get("plan", [])
    idx = state.get("plan_index", 0)

    if not plan or idx >= len(plan):
        state["next_action"] = "end"
        return state

    action = plan[idx]
    state["next_action"] = action
    state["plan_index"] = idx + 1
    state.setdefault("visited_actions", []).append(action)

    print(f"[Executor] Step {idx+1}/{len(plan)} → {action}")
    return state

# ====== Worker Nodes ======
def cv_inference(state: DermState) -> DermState:
    # giả sử đã kiểm tra image hợp lệ trước đó
    if state.get("image"):
        state["cv_results"] = cv_tool.invoke(state["image"])
    else:
        state["cv_results"] = {}
    print(f"[CV Inference] Results: {state['cv_results']}")
    return state

def llm_ask_symptom(state: DermState) -> DermState:
    """
    Nếu CV không chắc hoặc chưa có symptoms -> hỏi user thêm (và END luồng, chờ người dùng).
    Nếu không cần hỏi -> trả về cho executor tiếp tục.
    """
    need_ask = False
    if state.get("cv_results"):
        try:
            max_prob = max(state["cv_results"].values())
        except Exception:
            max_prob = 0.0
        if max_prob < 0.6:
            need_ask = True
    else:
        if not state.get("symptoms"):
            # không có kết quả CV và chưa có symptoms => hỏi
            need_ask = True

    if need_ask and not state.get("other_topic", False):
        q = llm.invoke(
            f"Kết quả hiện tại: {state.get('cv_results')}. "
            "Và bạn chưa có mô tả triệu chứng nào. "
            "Kết quả chưa chắc chắn. Hãy hỏi bệnh nhân thêm các triệu chứng (vị trí, thời gian, đau/không, lan không...)."
        )
        state["answer"] = q.content
        # state["awaiting_input"] = True
        # Khi hỏi user, dừng luồng, chờ user trả lời => END
        state["next_action"] = "end"
    else:
        # không cần hỏi -> tiếp tục plan (quay lại executor)
        state["next_action"] = "executor"  # sẽ được dùng trong conditional edges
    return state

# def symptom_matching(state: DermState) -> DermState:
#     if state.get("symptoms") or (state.get("user_input") and not state.get("other_topic", False)):
#         init_query = state.get("symptoms") or state.get("user_input")
#         query = init_query
#         if state['cv_results']:
#         #     # sắp xếp theo value giảm dần
#         #     top4 = sorted(state['cv_results'].items(), key=lambda x: x[1], reverse=True)[:4]
#         #     # lấy danh sách 4 key
#         #     top4_keys = [k for k, v in top4]
#         # for key in top4_keys:
#         #     query = init_query + f", {key}"
#         #     state["rag_docs"].append(knowledge_retrieval_tool(str(query)))
#         #     query = init_query
#             query = init_query + max(state["cv_results"], key=state["cv_results"].get)
#         state["rag_docs"] = knowledge_retrieval_tool.invoke(str(query))
#         print(f"[Symptom Matching] Retrieved {state['rag_docs']} docs for query: {query}")
#     else:
#         state["rag_docs"] = []
#     return state

def symptom_matching(state: DermState) -> DermState:
    if state.get("symptoms") or (state.get("user_input") and not state.get("other_topic", False)):
        init_query = state.get("symptoms") or state.get("user_input")
        query = str(init_query)  # Chuyển thành chuỗi
        if state.get('cv_results'):
            # Thay vì cộng chuỗi, chúng ta có thể làm query phức tạp hơn sau này
            # Nhưng hiện tại, chỉ cần query gốc là đủ tốt
            pass

        # Chỉ cần gọi tool với query
        state["rag_docs"] = knowledge_retrieval_tool.invoke(query)
        print(f"[Symptom Matching] Retrieved {len(state.get('rag_docs', []))} docs for query: {query}")
    else:
        state["rag_docs"] = []
    return state

def llm_reasoning(state: DermState) -> DermState:
    prompt = f"""
    Bạn là bác sĩ da liễu.
    - Kết quả CV: {state.get('cv_results')}
    - Triệu chứng: {state.get('symptoms') or state.get('user_input')}
    - Tài liệu RAG: {state.get('rag_docs')}
    Nhiệm vụ: chọn bệnh phù hợp nhất và giải thích ngắn gọn.
    """
    # # demo trả giá trị giả (ở production bạn nên gọi reasoner.invoke)
    # result = reasoner.invoke(prompt)
    # state["final_diagnosis"] = result.diagnosis
    # state["reasoning"] = result.reasoning
    # # state["final_diagnosis"] = "bệnh mẫu (demo)"
    # # state["reasoning"] = "vì triệu chứng phù hợp và kết quả tham khảo"
    # return state

    try:
        result = reasoner.invoke(prompt)
        state["final_diagnosis"] = result.diagnosis
        state["reasoning"] = result.reasoning
    except Exception as e:
        print(f"[LLM Reasoning] Error parsing structured output: {e}")
        # Cung cấp một giá trị dự phòng để graph có thể tiếp tục
        state["final_diagnosis"] = "Không thể xác định"
        state["reasoning"] = ("Đã có lỗi xảy ra trong quá trình phân tích chẩn đoán. "
                              "Thông tin đầu vào có thể chưa đủ hoặc không rõ ràng.")

    return state


# def llm_answer(state: DermState) -> DermState:
#     """
#     Luôn là bước kết thúc: trả lời tự nhiên cho bệnh nhân.
#     Sau khi thực hiện node này, luồng sẽ đi tới END.
#     """
#     # result = llm.invoke(f"""
#     # Với chẩn đoán: {state.get('final_diagnosis')}
#     # Lý do: {state.get('reasoning')}
#     # Viết câu trả lời tự nhiên cho bệnh nhân:
#     # - Giải thích bệnh
#     # - Triệu chứng điển hình
#     # - Gợi ý chăm sóc ban đầu
#     # - Nhắc đi khám nếu cần
#     # """)
#     prompt=f"""
#     # Với chẩn đoán: {state.get('final_diagnosis')}
#     # Lý do: {state.get('reasoning')}
#     # Viết câu trả lời tự nhiên cho bệnh nhân:
#     # - Giải thích bệnh
#     # - Triệu chứng điển hình
#     # - Gợi ý chăm sóc ban đầu
#     # - Nhắc đi khám nếu cần
#     # """
#     # state["answer"] = result.content
#     result = llm.invoke(prompt)
#     print("DEBUG LLM ANSWER RAW:", result)          # <== thêm dòng này
#     print("DEBUG LLM ANSWER CONTENT:", result.content if hasattr(result, "content") else None)
#
#     # fallback
#     if hasattr(result, "content") and result.content:
#         state["answer"] = result.content
#     else:
#         state["answer"] = str(result)   # fallback nếu content rỗng
#     # state["awaiting_input"] = False
#     state["next_action"] = "end"
#     return state


def llm_answer(state: DermState) -> DermState:
    """
    Luôn là bước kết thúc: trả lời tự nhiên cho bệnh nhân,
    BẮT BUỘC phải dựa trên thông tin từ RAG.
    """
    # Chuyển đổi list các Document thành một chuỗi context
    context_str = "\n\n---\n\n".join(
        [doc.page_content for doc in state.get('rag_docs', [])]
    )

    # Sửa lại prompt để "ground" câu trả lời vào context
    prompt_template = """Bạn là một trợ lý y tế chuyên về da liễu.
    Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách an toàn và chính xác, DỰA HOÀN TOÀN vào các tài liệu được cung cấp dưới đây.

    **QUY TẮC TUYỆT ĐỐI:**
    - KHÔNG được bịa đặt thông tin.
    - KHÔNG sử dụng kiến thức bên ngoài các tài liệu được cung cấp.
    - Nếu các tài liệu không chứa đủ thông tin để trả lời, hãy nói rằng "Dựa trên thông tin hiện có, tôi chưa thể kết luận chắc chắn về vấn đề này."

    **Tài liệu tham khảo (Context):**
    {context}

    **Chẩn đoán sơ bộ từ bước trước (để tham khảo):**
    - Chẩn đoán: {diagnosis}
    - Lý do: {reasoning}

    **Câu hỏi của người dùng:**
    {user_input}

    Dựa vào các quy tắc và thông tin trên, hãy viết một câu trả lời hoàn chỉnh, tự nhiên và thân thiện cho người dùng, bao gồm các gợi ý chăm sóc và nhấn mạnh việc cần đi khám bác sĩ.
    """

    prompt = prompt_template.format(
        context=context_str if context_str else "Không có tài liệu tham khảo.",
        diagnosis=state.get('final_diagnosis', 'Chưa xác định'),
        reasoning=state.get('reasoning', 'Không có'),
        user_input=state.get('user_input', '')
    )

    result = llm.invoke(prompt)

    if hasattr(result, "content") and result.content:
        state["answer"] = result.content
    else:
        state["answer"] = str(result)

    state["next_action"] = "end"
    return state


# ====== Build Graph ======
workflow = StateGraph(DermState)

# add nodes
workflow.add_node("validate", validate_input)
workflow.add_node("router", classify_topic)
workflow.add_node("planner", orchestrator_plan)
workflow.add_node("other_topic", other_topic_answer)
workflow.add_node("executor", executor)
workflow.add_node("cv_inference", cv_inference)
workflow.add_node("llm_ask_symptom", llm_ask_symptom)
workflow.add_node("symptom_matching", symptom_matching)
workflow.add_node("llm_reasoning", llm_reasoning)
workflow.add_node("llm_answer", llm_answer)

# START -> validate (validate luôn chạy đầu tiên)
workflow.add_edge(START, "validate")

# validate có thể dẫn tới router hoặc trả lời/END trực tiếp
workflow.add_conditional_edges(
    "validate",
    lambda s: s.get("next_action", "end"),
    {
        "router": "router",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "router",
    lambda s: s.get("next_action", "other_topic"),
    {
        "planner": "planner",
        "other_topic": "other_topic",
    },
)

# Nếu validate quyết định planner -> planner -> executor
workflow.add_edge("planner", "executor")

# Executor sẽ chuyển theo next_action trong plan
workflow.add_conditional_edges(
    "executor",
    lambda s: s.get("next_action", "end"),
    {
        "validate": "validate",
        "cv_inference": "cv_inference",
        "llm_ask_symptom": "llm_ask_symptom",
        "symptom_matching": "symptom_matching",
        "llm_reasoning": "llm_reasoning",
        "llm_answer": "llm_answer",
        "end": END,
    },
)

# Các worker không terminal -> quay về executor
for node in ["cv_inference", "symptom_matching", "llm_reasoning"]:
    workflow.add_edge(node, "executor")

# llm_ask_symptom: có thể dừng (END) hoặc tiếp tục (executor) => conditional
workflow.add_conditional_edges(
    "llm_ask_symptom",
    lambda s: s.get("next_action", "executor"),
    {
        "end": END,
        "executor": "executor",
    },
)

# llm_answer luôn kết thúc luồng hiện tại
workflow.add_edge("llm_answer", END)

# Compile graph
graph = workflow.compile()

async def run_derm_graph(user_question: str, image_url: str = None):
    # input cho graph
    # input_data = {"question": user_question, "image_url": image_url}
    input_data = {
        "user_input": user_question,
        "image": image_url,  
        "cv_results": None,
        "symptoms": None,
        "rag_docs": None,
        "final_diagnosis": None,
        "reasoning": None,
        "answer": None,
        "plan": None,
        "plan_index": 0,
        # "awaiting_input": False,
    }
    print(f"Running derm graph with input: {input_data}") 
    print()

    # chạy graph
    result = await graph.ainvoke(input_data)
    return result["answer"]

# # ====== Run graph (example) ======
# initial_state = {
#     "user_input": "Da bị đỏ và ngứa",
#     "image": "img/image.png",    # thử đổi None / path invalid để test branch
#     "cv_results": None,
#     "symptoms": None,
#     "rag_docs": None,
#     "final_diagnosis": None,
#     "reasoning": None,
#     "answer": None,
#     "plan": None,
#     "plan_index": 0,
    # "awaiting_input": False, 
# }


# thread = {"configurable": {"thread_id": "patient-001"}}

# for event in graph.stream(initial_state, thread, stream_mode="values"):
#     print("\n=== Step ===")
#     for k, v in event.items():
#         print(f"{k}: {v}")

if __name__ == "__main__":
    import asyncio

    # Ví dụ chạy thử khi thực thi file trực tiếp
    async def main():
        # Test case 1: Có ảnh và triệu chứng
        print("--- Running Test Case 1: Image and Symptoms ---")
        answer1 = await run_derm_graph(
            user_question="Da của tôi bị nổi mẩn đỏ và rất ngứa ở vùng cánh tay.",
            image_url="img/image.png"  # Thay bằng đường dẫn ảnh thật để test
        )
        print("\nFinal Answer 1:", answer1)

        # Test case 2: Chỉ có triệu chứng
        print("\n--- Running Test Case 2: Symptoms Only ---")
        answer2 = await run_derm_graph(
            user_question="Tôi bị ngứa ở da đầu và có vảy trắng.",
            image_url=None
        )
        print("\nFinal Answer 2:", answer2)

        # Test case 3: Hỏi ngoài chủ đề
        print("\n--- Running Test Case 3: Off-topic Question ---")
        answer3 = await run_derm_graph(
            user_question="Thủ đô của Việt Nam là gì?",
            image_url=None
        )
        print("\nFinal Answer 3:", answer3)

    # Chạy hàm main bất đồng bộ
    asyncio.run(main())