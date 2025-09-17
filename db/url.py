from scrape import scrape_website, clean_body_content, extract_body_content
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "vinmec_rag"
PERSIST_DIR = "./chroma_db"
TOP_TITLES = 3
NUM_RESULTS = 10

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def read_data_url(filename: str):
    list_url = []
    with open(filename, "r") as file:
        for line in file:
            url = line.strip()
            if url and url not in list_url:
                list_url.append(url)
    return list_url

def split_text_to_chunks(document_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_text(document_text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def store_chunks_in_chroma(chunks, vector_store, title, url, persist_dir=PERSIST_DIR):
    metadatas = [{"title": title, "url": url} for _ in chunks]
    ids = [f"{title}_{i}" for i in range(len(chunks))]

    if vector_store is None:
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_dir,
            metadatas=metadatas,
            ids=ids
        )
    else:
        if not chunks:
            return vector_store
        vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
    return vector_store

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
def retrieve_relevant_chunks(vector_store, titles, query_text, num_results=NUM_RESULTS):
    top_titles = get_top_titles(query_text, titles, top_k=TOP_TITLES)
    # print(f"[DEBUG] Closest titles: {top_titles}")
    number_titles = len(top_titles)
    all_docs = vector_store.similarity_search(query_text, k=number_titles*num_results)  
    filtered_docs = [doc for doc in all_docs if doc.metadata.get("title") in top_titles]

    seen = set()
    unique_chunks = []
    for doc in filtered_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_chunks.append(doc)
        if len(unique_chunks) >= num_results:
            break
    return unique_chunks

def load_data_to_chromaDB(url_file):
    url_list = read_data_url(url_file)

    # Xóa chroma cũ
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vector_store = None
    titles = []

    for url in url_list:
        print(f"Scraping url: {url}")
        html = scrape_website(url)
        body_content, title = extract_body_content(html)
        cleaned_content = clean_body_content(body_content)
        titles.append(title)
        chunks = split_text_to_chunks(cleaned_content)
        vector_store = store_chunks_in_chroma(chunks, vector_store, title, url)

    return vector_store, titles

vector_store, titles = load_data_to_chromaDB(url_file="db/data/url_data.csv")
# while True:
# user_input = input("Enter your query (or 'exit' to quit): ")
# if user_input.lower() == 'exit':
#     break
user_input = "Cách điều trị bệnh mụn nhọt là gì?"
query = user_input
chunks = retrieve_relevant_chunks(vector_store, titles, query)
print("-------------------------------------------------------------------------------------")
print(f"Query: {query}")
print("-------------------------------------------------------------------------------------")
print("Top titles:")
for title in titles:
    print(title)
print("--------------------------------------------------------------------------------------")
print("Top chunks:")
for chunk in chunks:
    print(chunk.page_content)
    print("-------------------------------------------------------------------------------------")  
