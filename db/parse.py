from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.1")

# template="Hãy phân tích nội dung sau:\n{dom_content}\n\nYêu cầu: {parse_description}"

template = """
Bạn là một agent trả lời dựa trên dữ liệu sau:

{dom_content}

Câu hỏi: {parse_description}

Yêu cầu: 
- Nếu trong context có thông tin, hãy trả lời đúng y như vậy.
- Nếu context không có, hãy trả lời: "Không tìm thấy thông tin trong cơ sở dữ liệu."
- Tuyệt đối không bịa thêm.
"""

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke({"dom_content": chunk, "parse_description": parse_description})
        parsed_results.append(response)

    return "\n".join(parsed_results)
