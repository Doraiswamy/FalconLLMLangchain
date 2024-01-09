import os

from flask import Flask, request, jsonify

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA

app = Flask(__name__)


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


# Load the chain globally once at application startup
chain = chainLoad()


@app.route("/chatbot", methods=["POST"])
def chainInference():
    query = request.json.get("query")
    if query:
        response = chain.run(query)  # Access the global chain variable
        return jsonify({"response": response})
    else:
        return jsonify({"error": "Query not provided"}), 400


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
