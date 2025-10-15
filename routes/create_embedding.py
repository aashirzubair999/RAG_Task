from flask import Blueprint, request, jsonify
from langchain_openai import OpenAIEmbeddings   #type: ignore
from langchain.document_loaders import PyPDFLoader #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  #type: ignore
from langchain.vectorstores import Chroma #type: ignore
import os
from dotenv import load_dotenv

load_dotenv()
Openai_Api_key = os.getenv("OPEN_API_KEY")

embedding_bp = Blueprint('embedding', __name__)

@embedding_bp.route("/createembedding", methods=["POST"])
def create_embedding():
    data = request.get_json()
    pdf_paths = data.get("pdf_paths", [])

    if not pdf_paths:
        return jsonify({"error": "No PDF paths provided"}), 400

    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=Openai_Api_key)
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="vectorstores/my_vector_db")

    vectordb.persist()
    vectordb = None

    return jsonify({
        "message": "Embeddings created successfully",
        "vector_db_path": "vectorstores/my_vector_db"
    })
