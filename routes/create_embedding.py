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

    # Create an empty list to hold all the documents from all PDFs
    docs = []

    # Loop through each PDF path provided in the request
    for path in pdf_paths:
        loader = PyPDFLoader(path)          # Load the PDF using LangChain's PyPDFLoader
        docs.extend(loader.load())          # Extract text (pages) from the PDF and add them to the docs list

    # Split large text documents into smaller chunks for better processing by the LLM
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)  # Each document is divided into smaller overlapping text chunks

    # Initialize the OpenAI embeddings model using your API key
    embeddings = OpenAIEmbeddings(openai_api_key=Openai_Api_key)

    # Create a vector database (Chroma) from the text chunks and their embeddings
    # persist_directory is where the vector database will be saved on disk
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="vectorstores/my_vector_db")

    # Save the vector database to disk so it can be used later for queries
    vectordb.persist()

    vectordb = None

    return jsonify({
        "message": "Embeddings created successfully",
        "vector_db_path": "vectorstores/my_vector_db"
    })
