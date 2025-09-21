vector_store = Chroma.from_texts(texts=chunks, 
                                     embedding=embedding_model, 
                                     collection_name=COLLECTION_NAME, 
                                     persist_directory=persist_dir)