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

query_type = ['qna']


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
                    I'll provide you with some context and history related to a question. Please carefully consider the information and answer the question thoughtfully. If the answer isn't found in the context, use your extensive knowledge and understanding of the world to provide a comprehensive response. ALso try to format the answer point wise where ever possible.
                    
                    Here are some RULES YOU NEED TO FOLLOW in order of importance:
                    1. If the user query contains phrases like "create incident", "open ticket", or "report a problem/incident", immediately respond "["trigger_form_create_incident"]" EXACTLY. Do not check or add it to the history or context.                   
                    2. If the user query mentions terms like "incident creation", "reporting procedure", "how to check the incident status" or "submitting a ticket", use the provided context and history to understand the specific information they need and provide a comprehensive response.
                    3. If the user query contains a word with alphanumeric characters (eg. INC20210100024) or phrases like "check an incident INC20210100024", "status of incident INC20210100024" or "INC20210100024", immediately respond "["trigger_form_check_incident","INC20210100024"]" EXACTLY AS A STRING. 
                    4. Give response in english only.

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

    llm = OpenAI(temperature=0.3, max_tokens=200)
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
        if 'trigger_form' in query_type[0]:
            response, response_type = conversationalChainInference(query, query_type)
            query_type = response_type
            return jsonify({'response': response})
        else:
            response = chain.run(query)
            print(response)
            if 'trigger_form' in response:
                response, response_type = conversationalChainInference(query, eval(response.strip())
)
                query_type = response_type
                return jsonify({'response': response})
            query_type = ['qna']
            return jsonify({'response': response})
    else:
        query_type = ['qna']
        return jsonify({'response': 'query not provided'}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
