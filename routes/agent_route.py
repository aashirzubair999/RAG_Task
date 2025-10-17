from flask import Blueprint, request, jsonify
from langchain.agents import Tool, initialize_agent, AgentType #type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   #type: ignore
from langchain.chains import RetrievalQA    #type: ignore
from langchain.vectorstores import Chroma   #type: ignore
from langchain.prompts import PromptTemplate #type: ignore
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
Open_Api_Key = os.getenv("OPEN_API_KEY")

agent_bp = Blueprint('agent', __name__)


# 1. Age calculator tool
def calculate_age(dob: str):
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return f"The age is {age} years."
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD."

age_tool = Tool(
    name="Age Calculator",
    func=calculate_age,
    description=(
        "Use this tool ONLY when user explicitly asks about AGE or provides a BIRTH DATE to calculate age. "
    )
)


# 2. Document QA Tool
def retrieval_qa_tool(query: str, db_path: str = "vectorstores/my_vector_db"):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=Open_Api_Key)
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Check if vector store has any documents
        if vectordb._collection.count() == 0:
            return "No documents have been uploaded yet. Please upload documents first."
        
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=Open_Api_Key, temperature=0)
    
        
        prompt_template = """You are a document Q&A assistant. Use ONLY the following context to answer the question.

STRICT RULES:
- If the answer is NOT in the context below, you MUST respond with: "I cannot find information about this in the uploaded documents."
- DO NOT use your general knowledge
- DO NOT make up answers
- ONLY use information from the context provided

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever, 
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa_chain.run(query)
        return result
    except Exception as e:
        return f"Error retrieving document information: {str(e)}"

retrieval_tool = Tool(
    name="Document QA",
    func=retrieval_qa_tool,
    description=(
        "Use this tool to search for information in UPLOADED DOCUMENTS ONLY. "
        "This tool searches document content about business plans, facilities, products, etc. "
        "If information is not in the documents, the tool will say so. "
        "Input: the user's question."
    )
)


# 3. AGENT SETUP with Custom Prompt
tools = [age_tool, retrieval_tool]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=Open_Api_Key)

# Custom prompt template that restricts the agent
PREFIX = """You are OsAMA, an intelligent assistant that helps users with their queries.

CRITICAL RULES:
- You can ONLY answer questions using your available tools
- You have TWO tools:
  1. Age Calculator - for calculating age from birth dates
  2. Document QA - for searching UPLOADED DOCUMENTS only
  
- The Document QA tool can ONLY answer questions about content in uploaded documents
- If Document QA returns "I cannot find information about this in the uploaded documents", you MUST tell the user that this information is not available in the documents
- DO NOT answer questions from general knowledge
- DO NOT provide information that is not from your tools

You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    max_execution_time=30,
    early_stopping_method="generate",
    agent_kwargs={
        'prefix': PREFIX,
        'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    }
)

print("Agent initialized successfully!")


@agent_bp.route("/agent", methods=["POST"])
def agent_route():
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        response = agent.invoke({"input": user_message})
        return jsonify({"response": response["output"]})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500