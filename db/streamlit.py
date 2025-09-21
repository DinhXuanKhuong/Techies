import streamlit as st
from scrape import scrape_website, split_dom_content, clean_body_content, extract_body_content

from parse import parse_with_ollama
from load_data_to_chromaDB import load_to_chromadb

st.title("Test AI Web Scraper")
url = st.text_input("Enter a website URL: ")

if st.button("Scrap site"):
    st.write("Scraping...")
    result = scrape_website(url)
    body_content, title = extract_body_content(result)
    cleaned_content = clean_body_content(body_content)
    st.session_state.dom_content = cleaned_content

    st.write(cleaned_content)
    print(cleaned_content)

if "dom_content" in st.session_state:
    parse_description = st.text_area("Describe what you want to parse?")

    if st.button("Parse Content"):
        if parse_description:
            st.write("Parsing the content")

        # raw_texts = split_dom_content(st.session_state.dom_content)
        raw_texts = st.session_state.dom_content
        print(raw_texts)
        load_to_chromadb(raw_texts)
        result = parse_with_ollama(parse_description)
        st.write(result)


