from flask import Blueprint, request, jsonify
from langchain.agents import Tool, initialize_agent, AgentType  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
import os
from dotenv import load_dotenv
from datetime import datetime, date
import re

load_dotenv()

class WeatherAgent:
    def __init__(self, api_key=None):
        """Initialize the WeatherAgent with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPEN_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)
        self.tools = self._initialize_tools()
    
    def age_calculator(self, dob_input, add_years=0):
        """Calculate age from date of birth input, optionally add years"""
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
            
            final_age = age + add_years
            return f"The age is {final_age} years."
        except ValueError as e:
            return f"Invalid date combination. {str(e)}. Please check that the day is valid for the given month and year."
    
    def _initialize_tools(self):
        age_tool = Tool(
            name="Age Calculator",
            func=lambda x: self.age_calculator(*eval(x) if x.startswith('(') else x),
            description=(
                "Use this tool when the user provides a COMPLETE date of birth with day, month, AND year, "
                "and optionally additional years to add. "
                "Valid formats: '15-08-2000', '15/08/2000', '(15, 8, 2000)', '15 August 2000', "
                "or '(15, 8, 2000, 5)' to add 5 years. "
                "DO NOT use this tool if only year is provided or if day/month is missing. "
                "The input must include all three: day, month, year, and optionally add_years."
            )
        )
        return [age_tool]
    
    def _create_agent_with_memory(self, memory):
        """Create agent instance with specific memory"""
        PREFIX = """You are OsAMA, an intelligent assistant that helps users with their queries.

CRITICAL RULES FOR OUTPUT FORMAT:
- You MUST ALWAYS end your response with "Final Answer: [your response]"
- Even for simple greetings or conversations, use the format: "Final Answer: [your greeting/response]"
- This is REQUIRED for the system to work properly

YOUR CAPABILITIES:
- You can answer questions using your available tools OR respond to social/conversational messages
- You have ONE tool: Age Calculator - for calculating age from birth dates (REQUIRES complete date: day, month, year, optional add_years)
  
AGE CALCULATOR RULES:
- You MUST have all three components (day, month, year) before using the tool
- If user only provides year (e.g., "2002"), ask for day and month
- If user only provides month and year, ask for day
- If user only provides day and month, ask for year
- If user provides month in number form like 01, 02, 3, 4, accept and process it
- Optionally, user can provide additional years to add (e.g., "add 5 to my age")
- DO NOT attempt to calculate age with incomplete information
  
SOCIAL INTERACTIONS (respond directly but ALWAYS use "Final Answer:" format):
- Greetings (hello, hi, hey) → Respond warmly and introduce yourself
- "How are you?" → Respond positively and offer help
- "What's up?" → Respond casually and ask how you can help
- "Thank you" → Acknowledge gratefully
- "Good morning/afternoon/evening" → Greet back warmly
- "Goodbye/Bye" → Say farewell politely
- "Who are you?" → Introduce yourself as OsAMA, an intelligent assistant
- Any friendly conversational messages → Respond warmly and naturally
- Questions about previous conversation → Use your chat history to answer

MEMORY AND CONTEXT:
- You have access to the full chat history provided in the request
- When users ask "what did I ask before" or "what did the last 5 questions did I ask from you" or similar, review the chat_history and provide accurate information
- Reference previous conversations naturally when relevant

LIMITATIONS:
- Do not answer general knowledge questions (e.g., "when was Pakistan created")
- For topics NOT related to age calculation, politely inform the user about your capabilities

You have access to the following tools:"""

        FORMAT_INSTRUCTIONS = """IMPORTANT: You must use this exact format for ALL responses:

Question: the input question you must answer
Thought: you should always think about what to do

For conversational/social messages (greetings, thanks, etc.):
Thought: This is a social interaction, I should respond directly
Final Answer: [your warm, friendly response]

For questions requiring tools:
Thought: I need to use a tool to answer this
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL: Every response MUST end with "Final Answer:" followed by your response."""

        SUFFIX = """Begin!

Chat History (review this for context about previous conversations):
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
            max_iterations=5,
            max_execution_time=30,
            early_stopping_method="generate",
            agent_kwargs={
                'prefix': PREFIX,
                'format_instructions': FORMAT_INSTRUCTIONS,
                'suffix': SUFFIX
            }
        )
        return agent
    
    def process_message(self, user_message, chat_history=None):
        """Process user message with provided chat history"""
        try:
            # Initialize memory with provided chat history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Load provided chat history into memory
            if chat_history:
                for entry in chat_history:
                    if entry.get("user"):
                        memory.chat_memory.add_user_message(entry["user"])
                    if entry.get("AI"):
                        memory.chat_memory.add_ai_message(entry["AI"])
            
            # Create agent with memory
            agent = self._create_agent_with_memory(memory)
            response = agent.invoke({"input": user_message})
            
            # Update chat history with new message and response
            new_history = chat_history.copy() if chat_history else []
            new_history.append({"user": user_message, "AI": response["output"]})
            
            return {"response": response["output"], "chat_history": new_history}, 200
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e), "chat_history": chat_history}, 500

weather_bp = Blueprint('weather', __name__)
weather_agent = WeatherAgent()

@weather_bp.route("/weather", methods=["POST"])
def weather_route():
    data = request.get_json()
    user_message = data.get("message")
    chat_history = data.get("chat_history", [])

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    result, status_code = weather_agent.process_message(user_message, chat_history)
    return jsonify(result), status_code