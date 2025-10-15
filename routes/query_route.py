from flask import Blueprint, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI   #type: ignore
from langchain.vectorstores import Chroma   #type: ignore
from langchain.chains import RetrievalQA    #type: ignore
import os
from dotenv import load_dotenv

load_dotenv()
Openai_Api_key = os.getenv("OPEN_API_KEY")

query_bp = Blueprint('query', __name__)

@query_bp.route("/query", methods=["POST"])
def query_route():
    data = request.get_json()
    db_path = data.get("db_path")
    user_query = data.get("query")

    if not db_path or not user_query:
        return jsonify({"error": "db_path and query are required"}), 400

    embeddings = OpenAIEmbeddings(openai_api_key=Openai_Api_key)
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=Openai_Api_key)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    response = qa_chain.run(user_query)

    return jsonify({"answer": response})
