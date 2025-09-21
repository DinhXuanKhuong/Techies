# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "vinmec_rag"
PERSIST_DIR = "./medical_db"
TOP_TITLES = 3
NUM_RESULTS = 5

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)



def get_top_titles(query_text, titles, top_k=TOP_TITLES):
    query_vec = embedding_model.embed_query(query_text)
    scores = []
    for title in set(titles):
        title_vec = embedding_model.embed_query(title)
        score = cosine_similarity([query_vec], [title_vec])[0][0]
        scores.append((title, score))
    top_titles = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [t[0] for t in top_titles]

# Ideas: Top relevant chunks -> filter chunks by top titles -> retrieve top chunks from filtered chunks
# def retrieve_relevant_chunks(vector_store, titles, query_text, num_results=NUM_RESULTS):
#     top_titles = get_top_titles(query_text, titles, top_k=TOP_TITLES)
#     # print(f"[DEBUG] Closest titles: {top_titles}")
#     number_titles = len(top_titles)
#     all_docs = vector_store.similarity_search(query_text, k=number_titles*num_results)
#     filtered_docs = [doc for doc in all_docs if doc.metadata.get("title") in top_titles]
#
#     seen = set()
#     unique_chunks = []
#     for doc in filtered_docs:
#         if doc.page_content not in seen:
#             seen.add(doc.page_content)
#             unique_chunks.append(doc)
#         if len(unique_chunks) >= num_results:
#             break
#     return unique_chunks

def load_titles_from_chroma(vector_store):
    # lấy tất cả documents trong DB
    docs = vector_store.get(include=["metadatas"])
    titles = set()
    for meta in docs["metadatas"]:
        if meta and "title" in meta:
            titles.add(meta["title"])
    return list(titles)


vector_store = Chroma(
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR
)
retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})

def retrieve_relevant_chunks(query_text: str) -> List[Document]:
    """
    Hàm mới, sử dụng retriever để lấy tài liệu.
    Hàm này sẽ thay thế hoàn toàn hàm cũ.
    """
    print(f"[RAG] Retrieving documents for query: {query_text}")
    # Chỉ cần gọi retriever.invoke là đủ
    return retriever.invoke(query_text)

# Load titles từ metadata
titles = load_titles_from_chroma(vector_store)

# # Giờ có thể dùng tiếp retrieve_relevant_chunks như cũ
# query = "Cách điều trị bệnh mụn nhọt là gì?"
# chunks = retrieve_relevant_chunks(vector_store, titles, query)
# for chunk in chunks:
#     print(chunk.page_content)