import os
from dotenv import load_dotenv

from flask import Flask, request, jsonify

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from conversatinal_langchain import conversationalChainInference

load_dotenv()

app = Flask(__name__)

query_type = 'qna'


def chainLoad():
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    with open("necdataset/mies.txt", encoding='utf-8') as f:
        state_of_the_union = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20)
    chunks = text_splitter.split_text(state_of_the_union)
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(chunks, embeddings)
    template = """
                    I'll provide you with some context and history related to a question. Please carefully consider the information and answer the question thoughtfully. If the answer isn't found in the context, use your extensive knowledge and understanding of the world to provide a comprehensive response.
                    
                    Here are some RULES YOU NEED TO FOLLOW in order of importance:
                    1. If the user query explicitly mentions any request to create an incident like for eg. 'create incident', return the word 'create_incident_form' exactly without checking the history or context.
                    2. If the user query is about information related to incident creation like for eg. 'how to create an incident', provide a comprehensive response.

                    Context:
                    {context}

                    History:
                    {history}

                    {question}
                    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    llm = OpenAI(temperature=0.7, max_tokens=150)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorStore.as_retriever(),
                                        chain_type_kwargs={
                                            "verbose": True,
                                            "prompt": prompt,
                                            "memory": ConversationBufferWindowMemory(
                                                k=5,
                                                memory_key="history",
                                                input_key="question"),
                                        })
    return chain


# Load the chain globally once at application startup
chain = chainLoad()


@app.route("/chatbot", methods=["POST"])
def chainInference():
    global query_type
    query = request.json.get('prompt')
    if query:
        if query_type == 'form':
            response, response_type = conversationalChainInference(query)
            query_type = response_type
            return jsonify({'response': response})
        else:
            response = chain.run(query)
            if 'create_incident_form' in response:
                response, response_type = conversationalChainInference(query)
                query_type = response_type
                return jsonify({'response': response})
            return jsonify({'response': response, 'query_type': 'qna'})
    else:
        query_type = 'qna'
        return jsonify({'response': 'query not provided'}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
