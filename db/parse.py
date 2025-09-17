from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from chromadb.utils import embedding_functions

model = OllamaLLM(model="llama3.1")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text"
)

template = """
Bạn là một agent trả lời dựa trên dữ liệu sau:

{context}

Câu hỏi: {question}

Yêu cầu: 
- Nếu trong dom_content có thông tin, hãy trả lời đúng y như vậy.
- Nếu dom_content không có, hãy trả lời: "Không tìm thấy thông tin trong cơ sở dữ liệu."
- Tuyệt đối không bịa thêm.
"""

def parse_with_ollama(question):
    prompt = ChatPromptTemplate.from_template(template)

    chroma_client = chromadb.PersistentClient(path="./medical_db")
    collection = chroma_client.get_or_create_collection(
        name="medical",
        embedding_function=ollama_ef
    )

    # chain = prompt | model
    results = collection.query(query_texts=[question], n_results=3)
    retrieved_docs = "\n".join(results["documents"][0]) if results["documents"] else ""

    print(retrieved_docs)
    chain = prompt | model
    response = chain.invoke({
        "context": retrieved_docs, 
        "question": question
    })

    return response
