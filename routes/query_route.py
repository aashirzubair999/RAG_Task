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

    # Create embeddings object using OpenAI — this converts text into numerical vectors
    embeddings = OpenAIEmbeddings(openai_api_key=Openai_Api_key)

    # Load the existing Chroma vector database where PDF embeddings are stored
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Convert the Chroma database into a retriever to fetch relevant text chunks
    retriever = vectordb.as_retriever()

    # Initialize the ChatGPT model (GPT-4o-mini) to generate answers
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=Openai_Api_key)

    # Create a Retrieval-QA chain that connects the retriever and the LLM
    # "stuff" means all retrieved text chunks are passed directly into the model's prompt
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Run the question-answering chain with the user's query and get the response
    response = qa_chain.run(user_query)

    return jsonify({"answer": response})
