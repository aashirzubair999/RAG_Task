from flask import Blueprint, request, jsonify, session
from langchain.agents import Tool, initialize_agent, AgentType  # type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
from langchain.chains import RetrievalQA  # type: ignore
from langchain.vectorstores import Chroma  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
import os
from dotenv import load_dotenv
from datetime import datetime, date
import re

load_dotenv()

# 🔹 Store memory separately for each user
user_memories = {}

def get_user_memory(user_id):
    """Return existing memory for a user or create a new one"""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    return user_memories[user_id]


class WeatherAgent:
    def __init__(self, api_key=None):
        """Initialize the WeatherAgent with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPEN_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)
        self.tools = self._initialize_tools()
        self.agent = None  # Will be created with user-specific memory
    
    def age_calculator(self, dob_input):
        """Calculate age from date of birth input"""
        if isinstance(dob_input, str):
            dob_input = dob_input.strip()
            match = re.findall(r'\d+', dob_input)
            
            if len(match) == 0:
                return "No date values found. Please provide date of birth as: day, month, year. Example: (15, 8, 2000)"
            elif len(match) == 1:
                return "Incomplete date. You provided only 1 value. Please provide all three: day, month, and year. Example: (15, 8, 2000)"
            elif len(match) == 2:
                return "Incomplete date. You provided only 2 values. Please provide all three: day, month, and year. Example: (15, 8, 2000)"
            elif len(match) == 3:
                dob_tuple = tuple(map(int, match))
            else:
                return f"Too many values ({len(match)} numbers found). Please provide exactly 3 values: day, month, year. Example: (15, 8, 2000)"
        elif isinstance(dob_input, (tuple, list)):
            if len(dob_input) < 3:
                missing = []
                if len(dob_input) < 1:
                    missing.append("day")
                if len(dob_input) < 2:
                    missing.append("month")
                if len(dob_input) < 3:
                    missing.append("year")
                return f"Incomplete date. Missing: {', '.join(missing)}. Please provide all three values."
            
            dob_tuple = dob_input
            
            if dob_tuple[0] is None:
                return "Day is missing in the date of birth. Please provide the day."
            if dob_tuple[1] is None:
                return "Month is missing in the date of birth. Please provide the month."
            if dob_tuple[2] is None:
                return "Year is missing in the date of birth. Please provide the year."
        else:
            return "Date of birth must be given as a tuple like (day, month, year)."
        
        if len(dob_tuple) != 3:
            return "Date of birth must have exactly 3 values: day, month, year."

        day, month, year = dob_tuple

        if not (1 <= day <= 31):
            return f"Invalid day: {day}. Day must be between 1 and 31."
        if not (1 <= month <= 12):
            return f"Invalid month: {month}. Month must be between 1 and 12."
        if year < 1900 or year > date.today().year:
            return f"Invalid year: {year}. Year must be between 1900 and {date.today().year}."

        try:
            birth_date = date(year, month, day)
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            if age < 0:
                return "Invalid date: The birth date is in the future."
            
            return f"The age is {age} years."
        except ValueError as e:
            return f"Invalid date combination. {str(e)}. Please check that the day is valid for the given month and year."
    
    def _initialize_tools(self):
        age_tool = Tool(
            name="Age Calculator",  
            func=self.age_calculator,
            description=(
                "Use this tool ONLY when the user provides a COMPLETE date of birth with day, month, AND year. "
                "Valid formats: '15-08-2000', '15/08/2000', '(15, 8, 2000)', '15 August 2000'. "
                "DO NOT use this tool if only year is provided (e.g., '2002') or if day/month is missing. "
                "If the date is incomplete, ask the user for the missing information. "
                "The input must include all three: day, month, year."
            )
        )
        return [age_tool]
    
    def _create_agent_with_memory(self, memory):
        """Create agent instance with specific user memory"""
        PREFIX = """You are OsAMA, an intelligent assistant that helps users with their queries.

CRITICAL RULES:
- You can answer questions using your available tools OR respond to social/conversational messages directly
- You have TWO tools:
  1. Age Calculator - for calculating age from birth dates (REQUIRES complete date: day, month, year)
  2. Document QA - for searching UPLOADED DOCUMENTS only
  
- For Age Calculator: You MUST have all three components (day, month, year) before using the tool
  * If user only provides year (e.g., "2002"), ask for day and month
  * If user only provides month and year, ask for day
  * If user only provides day and month, ask for year
  * if the user provide month in number form like 01 02 3 4 answer them as well
  * DO NOT attempt to calculate age with incomplete information
  
- For Social Interactions (respond directly WITHOUT using tools):
  * Greetings (hello, hi, hey) → Respond warmly and introduce yourself
  * "How are you?" → Respond positively and offer help
  * "What's up?" → Respond casually and ask how you can help
  * "Thank you" → Acknowledge gratefully
  * "Good morning/afternoon/evening" → Greet back warmly
  * "Goodbye/Bye" → Say farewell politely
  * "Who are you?" → Introduce yourself as OsAMA, an intelligent assistant
  * Any friendly conversational messages → Respond warmly and naturally

- Do not answer general questions like when pajsitan was created like these type of questions       

- The Document QA tool can ONLY answer questions about content in uploaded documents
- If Document QA returns "I cannot find information about this in the uploaded documents", you MUST tell the user that this information is not available in the documents
- For topics NOT related to age calculation or uploaded documents, politely inform the user about your capabilities

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

Chat History:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

        agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
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
        return agent
    
    def process_message(self, user_message, user_memory):
        """Process user message with specific user memory"""
        try:
            # Create or update agent with user's memory
            agent = self._create_agent_with_memory(user_memory)
            response = agent.invoke({"input": user_message})
            return {"response": response["output"]}, 200
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}, 500


weather_bp = Blueprint('weather', __name__)
weather_agent = WeatherAgent()


@weather_bp.route("/weather", methods=["POST"])
def weather_route():
    data = request.get_json()
    user_message = data.get("message")
    user_id = data.get("user_id")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    if not user_id:
        return jsonify({"error": "user_id is required for tracking session"}), 400

    # Get or create memory for this user
    user_memory = get_user_memory(user_id)

    result, status_code = weather_agent.process_message(user_message, user_memory)
    return jsonify(result), status_code


@weather_bp.route("/weather/clear", methods=["POST"])
def clear_memory_route():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id or user_id not in user_memories:
        return jsonify({"error": "Invalid or missing user_id"}), 400

    user_memories[user_id].clear()
    return jsonify({"message": f"Memory cleared for user {user_id}"}), 200