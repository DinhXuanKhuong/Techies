
import os
from dotenv import load_dotenv
from typing import TypedDict, Dict, List, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomaly import test_image
from cv_tool import predict_image, load_model

# ====== Load ENV ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====== Init LLM ======
llm = ChatOpenAI(
    temperature=0,
    api_key=OPENAI_API_KEY,
    model="qwen/qwen3-next-80b-a3b-thinking",
    base_url="https://openrouter.ai/api/v1",
    max_tokens=1024
)

# ====== Init models ======
cv_model, cv_device = load_model('best_model_final.pth')
anomaly_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
anomaly_model = torch.load(
    'autoencoder_skin.pth',
    map_location=anomaly_device,
    weights_only=False
)
anomaly_model.to(anomaly_device)
anomaly_model.eval()

criterion = nn.MSELoss()
threshold = 0.25
transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ====== Tools ======
@tool
def anomaly_tool(image_path: str) -> dict:
    """Kiểm tra ảnh có phải tổn thương da không."""
    error, is_anomaly = test_image(
        anomaly_model, image_path, transform,
        criterion, anomaly_device, threshold
    )
    return {"reconstruction_error": error, "is_anomaly": is_anomaly}

@tool
def cv_tool(image_path: str) -> dict:
    """Dự đoán bệnh từ ảnh da."""
    return predict_image(cv_model, image_path, cv_device)

@tool
def knowledge_retrieval_tool(query: str, context: str = None) -> List[str]:
    """Truy xuất thông tin y tế từ KB (mock)."""
    return [f"Mock thông tin cho {query}", "Nguồn: Bộ Y Tế VN"]

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
    awaiting_input: Optional[bool]   # nếu agent đang chờ user trả lời

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
    """
    Luôn chạy ngay sau START.
    Quyết định:
    - Nếu có ảnh: check anomaly. Nếu ảnh hợp lệ -> đi planner (planner có thể chứa cv_inference).
                Nếu ảnh không hợp lệ:
                    - Nếu đã có triệu chứng -> vẫn cho planner dựa trên symptoms (không dùng cv_inference).
                    - Nếu không có triệu chứng -> trả lời ngay (yêu cầu ảnh khác/triệu chứng) -> END.
    - Nếu không có ảnh:
        - Nếu có triệu chứng -> planner (dựa trên triệu chứng)
        - Nếu không -> trả lời yêu cầu thêm thông tin -> END
    """
    state["awaiting_input"] = False
    img = state.get("image")
    if img:
        try:
            result = anomaly_tool(img)
        except Exception as e:
            # nếu tool lỗi, ta coi như ảnh không khả dụng
            result = {"reconstruction_error": None, "is_anomaly": True}
        state["anomaly_check"] = result

        if result.get("is_anomaly") is True:
            # ảnh không hợp lệ cho chẩn đoán da
            if state.get("symptoms") or state.get("user_input"):
                # có triệu chứng → vẫn tiếp tục planner nhưng KHOẢNG CV sẽ bị bỏ qua bởi planner logic
                state["next_action"] = "planner"
            else:
                # KHÔNG có triệu chứng → trả lời yêu cầu gửi ảnh/miêu tả rồi END
                q = llm.invoke(
                    "Ảnh bạn gửi có vẻ không phải ảnh da có thể chẩn đoán. "
                    "Vui lòng gửi ảnh da rõ ràng hơn hoặc mô tả triệu chứng (ví dụ: 'vị trí, thời gian, đau/ít ngứa/không')."
                )
                state["answer"] = q.content
                state["awaiting_input"] = True
                state["next_action"] = "end"
        else:
            # ảnh hợp lệ
            state["next_action"] = "planner"
    else:
        # không có ảnh
        if state.get("symptoms") or state.get("user_input"):
            state["next_action"] = "planner"
        else:
            q = llm.invoke(
                "Bạn chưa gửi ảnh và cũng không miêu tả triệu chứng rõ ràng. "
                "Vui lòng gửi ảnh hoặc mô tả triệu chứng để tôi hỗ trợ."
            )
            state["answer"] = q.content
            state["awaiting_input"] = True
            state["next_action"] = "end"

    return state

# ====== Planner Node ======
def orchestrator_plan(state: DermState) -> DermState:
    """
    Tạo plan dựa trên kết quả validate:
    - Nếu ảnh có và hợp lệ -> bao gồm cv_inference
    - Nếu ảnh có nhưng không hợp lệ -> không có cv_inference (dùng symptom matching nếu có)
    - Nếu ảnh có và hợp lệ + symptoms -> nên hỏi thêm triệu chứng (llm_ask_symptom)
    - Nếu không ảnh nhưng có symptoms -> plan từ symptoms
    - Nếu không ảnh và không symptoms -> plan rỗng (validate đã đặt next_action=end)
    """
    plan = []
    img = state.get("image")
    anomaly = state.get("anomaly_check")
    # determine image_valid: anomaly exists and is False
    image_valid = False
    if img:
        if anomaly is None:
            # (cần thiết validate luôn chạy trước nên anomaly thường không None)
            image_valid = True
        else:
            image_valid = not anomaly.get("is_anomaly", True)

    if img and image_valid:
        # hình hợp lệ -> dùng CV rồi hỏi thêm triệu chứng nếu cần
        plan = [
            "cv_inference",
            "llm_ask_symptom",      # có thể dừng chờ user nếu LLM hỏi
            "symptom_matching",
            "llm_reasoning",
            "llm_answer"
        ]
    else:
        # không có ảnh hợp lệ
        if state.get("symptoms") or state.get("user_input"):
            plan = [
                "symptom_matching",
                "llm_reasoning",
                "llm_answer"
            ]
        else:
            # không có gì -> planner trả về rỗng; validate đã đặt next_action=end
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
        state["cv_results"] = cv_tool(state["image"])
    else:
        state["cv_results"] = {}
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

    if need_ask:
        q = llm.invoke(
            f"Kết quả hiện tại: {state.get('cv_results')}. "
            "Kết quả chưa chắc chắn. Hãy hỏi bệnh nhân thêm các triệu chứng (vị trí, thời gian, đau/không, lan không...)."
        )
        state["answer"] = q.content
        state["awaiting_input"] = True
        # Khi hỏi user, dừng luồng, chờ user trả lời => END
        state["next_action"] = "end"
    else:
        # không cần hỏi -> tiếp tục plan (quay lại executor)
        state["next_action"] = "executor"  # sẽ được dùng trong conditional edges
    return state

def symptom_matching(state: DermState) -> DermState:
    if state.get("symptoms") or state.get("user_input"):
        query = state.get("symptoms") or state.get("user_input")
        state["rag_docs"] = knowledge_retrieval_tool(str(query))
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
    # demo trả giá trị giả (ở production bạn nên gọi reasoner.invoke)
    # result = reasoner.invoke(prompt)
    # state["final_diagnosis"] = result.diagnosis
    # state["reasoning"] = result.reasoning
    state["final_diagnosis"] = "bệnh mẫu (demo)"
    state["reasoning"] = "vì triệu chứng phù hợp và kết quả tham khảo"
    return state

def llm_answer(state: DermState) -> DermState:
    """
    Luôn là bước kết thúc: trả lời tự nhiên cho bệnh nhân.
    Sau khi thực hiện node này, luồng sẽ đi tới END.
    """
    # result = llm.invoke(f"""
    # Với chẩn đoán: {state.get('final_diagnosis')}
    # Lý do: {state.get('reasoning')}
    # Viết câu trả lời tự nhiên cho bệnh nhân:
    # - Giải thích bệnh
    # - Triệu chứng điển hình
    # - Gợi ý chăm sóc ban đầu
    # - Nhắc đi khám nếu cần
    # """)
    prompt=f"""
    # Với chẩn đoán: {state.get('final_diagnosis')}
    # Lý do: {state.get('reasoning')}
    # Viết câu trả lời tự nhiên cho bệnh nhân:
    # - Giải thích bệnh
    # - Triệu chứng điển hình
    # - Gợi ý chăm sóc ban đầu
    # - Nhắc đi khám nếu cần
    # """
    # state["answer"] = result.content
    result = llm.invoke(prompt)
    print("DEBUG LLM ANSWER RAW:", result)          # <== thêm dòng này
    print("DEBUG LLM ANSWER CONTENT:", result.content if hasattr(result, "content") else None)

    # fallback
    if hasattr(result, "content") and result.content:
        state["answer"] = result.content
    else:
        state["answer"] = str(result)   # fallback nếu content rỗng
    state["awaiting_input"] = False
    state["next_action"] = "end"
    return state

# ====== Build Graph ======
workflow = StateGraph(DermState)

# add nodes
workflow.add_node("validate", validate_input)
workflow.add_node("planner", orchestrator_plan)
workflow.add_node("executor", executor)
workflow.add_node("cv_inference", cv_inference)
workflow.add_node("llm_ask_symptom", llm_ask_symptom)
workflow.add_node("symptom_matching", symptom_matching)
workflow.add_node("llm_reasoning", llm_reasoning)
workflow.add_node("llm_answer", llm_answer)

# START -> validate (validate luôn chạy đầu tiên)
workflow.add_edge(START, "validate")

# validate có thể dẫn tới planner hoặc trả lời/END trực tiếp
workflow.add_conditional_edges(
    "validate",
    lambda s: s.get("next_action", "end"),
    {
        "planner": "planner",
        "symptom_matching": "symptom_matching",
        "llm_answer": "llm_answer",
        "end": END,
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
    input_data = {"question": user_question, "image_url": image_url}

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
#     "awaiting_input": False,
# }

# thread = {"configurable": {"thread_id": "patient-001"}}

# for event in graph.stream(initial_state, thread, stream_mode="values"):
#     print("\n=== Step ===")
#     for k, v in event.items():
#         print(f"{k}: {v}")
