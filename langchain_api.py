import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA


def chainLoad():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    with open("necdataset/mies.txt") as f:
        state_of_the_union = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20)
    chunks = text_splitter.split_text(state_of_the_union)
    embeddings = HuggingFaceEmbeddings()
    vectorStore = FAISS.from_texts(chunks, embeddings)
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.5, "max_length": 1000})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorStore.as_retriever())
    return chain


def chainInference():
    st.title("NEC LLM BOT")
    if 'chain' not in st.session_state:
        st.session_state.chain = chainLoad()
    query = st.text_input("You:")
    if st.button("Send"):
        st.text("Chatbot: " + st.session_state.chain.run(query))


if __name__ == "__main__":
    chainInference()
