from flask import Blueprint, request, jsonify
from langchain.agents import initialize_agent, AgentType #type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings #type: ignore
from langchain.memory import ConversationBufferMemory #type: ignore
from langchain_core.tools import BaseTool #type: ignore
from langchain.chains import RetrievalQA #type: ignore
from langchain_community.vectorstores import FAISS #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
from langchain.docstore.document import Document #type: ignore
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from datetime import date
import re
import requests
import json

load_dotenv()

weather_bp = Blueprint('weather', __name__)


# Input schema for Weather Tool
class WeatherToolInput(BaseModel):
    query: str = Field(
        description=(
            "The complete user query about weather. "
            "Examples: 'tell me weather in London for next 5 days', "
            "'what's the weather in Paris', 'weather forecast for New York 3 days'. "
            "The tool will automatically extract the location and number of days from this query."
        )
    )


# Weather Tool with Vector Store
class WeatherTool(BaseTool):
    name: str = "Weather Forecast"
    args_schema: type[BaseModel] = WeatherToolInput
    return_direct: bool = False
    description: str = (
        "Mandatory tool-Use this tool when the user asks about weather, weather forecast, or climate conditions for a specific location. "
        "This can include incomplete strings like 'weather in London', 'forecast for Paris', 'tell me weather in Tokyo for 3 days'."
        "DO NOT answer age questions yourself - ALWAYS call this tool first. "
        "The tool will detect missing information and provide appropriate responses. "
        "a stringify object which will have 2 keys named as Days, Location "
        "Days and Location is provided by user. Is user missed or skip anything make sure to give the value to location key as NAN and days to 1 "
        "The tool accepts queries like: 'weather in London for 5 days', 'what's the weather in Paris', "
        "'forecast for New York'. It will automatically extract the city name and number of days. "
        "If days are not mentioned, it provides today's weather."
        "assume Location value if not provided make sure to give the value NAN to that particular key(Location). For example the JSON should be like"
        "('day':'day given by user or 1 if not given', 'Location':'location given by user or NAN if not given') "
        " Make sure to use the curly braces for JSON not the square braces i gave in example its just for your understanding"
    )
    
    def _extract_location_and_days(self, query: str) -> tuple:
        """Extract location and number of days from the query"""
        query_lower = query.lower()
        
        # Extract number of days
        days = 1  # default to today
        day_patterns = r'(\d+)\s*days?'  # Matches "3 days" or "3days"

        match = re.search(day_patterns, query_lower)
        if match:
            days = int(match.group(1))
        else:
            days = 1  # default value if no number of days is found

        
        # Extract location (city name)
        # Remove common weather-related words to isolate the city name
        cleaned_query = re.sub(r'\b(weather|forecast|tell|me|in|for|the|next|days?|about|what|whats|is)\b', '', query_lower)
        cleaned_query = re.sub(r'\d+', '', cleaned_query)  # Remove numbers
        cleaned_query = cleaned_query.strip()
        print ("Cleaned Query for Location Extraction:", cleaned_query)
        
        # Use cleaned query directly as the city name
        location = cleaned_query if cleaned_query else "Islamabad"  # Default to Islamabad

        return location.strip(), min(days, 7)  # Limit to 7 days max
    
    def _fetch_weather_data(self, location: str, days: int) -> str:
        """Fetch weather data from WeatherAPI"""
        api_key = os.getenv("WEATHER_API_KEY")
        
        if not api_key:
            return "Weather API key not found. Please add WEATHER_API_KEY to your .env file. Get free API key from https://www.weatherapi.com/"
        
        try:
            # This is the weblink from where we have get the api and will send the reuqest to it
            url = f"http://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": api_key,
                "q": location,
                "days": days
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return json.dumps(response.json(), indent=2)
        
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
    

    def _create_vector_store(self, weather_data: str, location: str) -> FAISS:
        """Create vector store from weather data"""
        try:
            # Parse the JSON data
            data = json.loads(weather_data)
          
            # Create documents from weather data
            documents = []
            
            # Current weather
            if "current" in data:
                current = data["current"]
                doc_text = f"""Current Weather in {location}:
                    Temperature: {current.get('temp_c', 'N/A')}°C ({current.get('temp_f', 'N/A')}°F)
                    Condition: {current.get('condition', {}).get('text', 'N/A')}
                    Feels Like: {current.get('feelslike_c', 'N/A')}°C
                    Humidity: {current.get('humidity', 'N/A')}%
                    Wind Speed: {current.get('wind_kph', 'N/A')} km/h
                    Wind Direction: {current.get('wind_dir', 'N/A')}
                    Pressure: {current.get('pressure_mb', 'N/A')} mb
                    Precipitation: {current.get('precip_mm', 'N/A')} mm
                    UV Index: {current.get('uv', 'N/A')}
                    Visibility: {current.get('vis_km', 'N/A')} km"""
                documents.append(Document(page_content=doc_text, metadata={"type": "current", "location": location}))
            
            # Forecast data
            if "forecast" in data and "forecastday" in data["forecast"]:
                for day_data in data["forecast"]["forecastday"]:
                    date = day_data.get("date", "N/A")
                    day_info = day_data.get("day", {})
                    
                    doc_text = f"""Weather Forecast for {location} on {date}:
                    Max Temperature: {day_info.get('maxtemp_c', 'N/A')}°C ({day_info.get('maxtemp_f', 'N/A')}°F)
                    Min Temperature: {day_info.get('mintemp_c', 'N/A')}°C ({day_info.get('mintemp_f', 'N/A')}°F)
                    Average Temperature: {day_info.get('avgtemp_c', 'N/A')}°C
                    Condition: {day_info.get('condition', {}).get('text', 'N/A')}
                    Chance of Rain: {day_info.get('daily_chance_of_rain', 'N/A')}%
                    Total Precipitation: {day_info.get('totalprecip_mm', 'N/A')} mm
                    Max Wind Speed: {day_info.get('maxwind_kph', 'N/A')} km/h
                    Average Humidity: {day_info.get('avghumidity', 'N/A')}%
                    UV Index: {day_info.get('uv', 'N/A')}
                    Sunrise: {day_data.get('astro', {}).get('sunrise', 'N/A')}
                    Sunset: {day_data.get('astro', {}).get('sunset', 'N/A')}"""
                    documents.append(Document(page_content=doc_text, metadata={"type": "forecast", "date": date, "location": location}))
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
            vector_store = FAISS.from_documents(split_docs, embeddings)
            
            return vector_store
        
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def _run(self, query: str) -> str:
        """Execute the weather tool"""
        try:
            # Check if query is a JSON string
            if query.strip():
                try:
                    # Parse JSON input
                    data = json.loads(query.replace("'", '"'))  # Replace single quotes with double quotes
                    print("Parsed JSON Data:", data)
                    location = data.get('Location', 'NAN')
                    days = int(data.get('Days', 1))
                    
                    # Handle NAN location
                    if location == 'NAN' or not location:
                        return "I need to know the location to fetch weather data. Please specify a city name."
                    
                except json.JSONDecodeError:
                    # If JSON parsing fails, fall back to text extraction
                    location, days = self._extract_location_and_days(query)
            else:
                # Extract location and days from plain text query
                location, days = self._extract_location_and_days(query)
            
            # Fetch weather data
            weather_data = self._fetch_weather_data(location, days)
            
            if weather_data.startswith("Error") or weather_data.startswith("Weather API"):
                return weather_data
            
            # Create vector store
            vector_store = self._create_vector_store(weather_data, location)
            
            # Create RetrievalQA chain
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPEN_API_KEY"))
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=False
            )
            
            # Query the vector store
            response = qa_chain.invoke({"query": f"What is the weather in {location} for {days} days?"})
            
            return response["result"]
        
        except Exception as e:
            return f"Error processing weather request: {str(e)}"




class AgeCalculatorInput(BaseModel):
    dob_input: str = Field(
        description=(
            "The complete user input about date of birth or age calculation, exactly as provided by the user. "
            "Pass the raw input even if incomplete. Examples: 'I was born in 2002', 'my birthday is July', "
            "'15 August', '2002', 'calculate my age'. The tool will handle all validation and missing data."
        )
    )


# Age Calculator Tool as a class
class AgeCalculatorTool(BaseTool):
    name: str = "Age Calculator"
    args_schema: type[BaseModel] = AgeCalculatorInput
    return_direct: bool = False # Do not send the tool’s answer directly to the user.
    description: str = (
        "**MANDATORY TOOL** -Use this tool when the user wants to calculate his age. The input to the tool should always be a JSON object send as"
        "This includes incomplete inputs like 'July 1999', '2002', 'born in August', etc. "
        "DO NOT answer age questions yourself - ALWAYS call this tool first. "
        "Pass the EXACT user input as dob_input without any interpretation. "
        "The tool will detect missing information and provide appropriate responses. "
        "Examples: 'I was born in 2002', 'July 15', 'calculate my age', '24/7/2002'."
        " a stringify object which will have 3 keys named as day, month, year and the values should be string but a valid day,"
        " month and year provided by user. Is user missed or skip anything make sure to give the value of that key as NAN. "
        "For example user only provide year as 2002 and skip day and month then the value of day and month should be NAN "
        " and year should be '2002'. Make sure to follow the instructions of JSON object. Never change the keys name. Never "
        "assume any value if not provided make sure to give the value NAN to that particular key. For example the JSON should be like"
        " ('day':'day given by user or NAN if not given', 'month':'month given by user or NAN if not given', 'year':'year given by user or NAN if not given', ) "
        " Make sure to use the curly braces for JSON not the square braces i gave in example its just for your understanding"
    )
    
    def _run(self, dob_input: str) -> str:
        """Calculate age from date of birth input"""
        if isinstance(dob_input, str):
            dob_input = dob_input.strip().lower()
            
            # Month name to number mapping
            month_map = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            # Extract all numbers from the string
            numbers = re.findall(r'\d+', dob_input)
            
            # Check if any month name is present
            month_from_name = None
            month_name_found = None
            for month_str, month_num in month_map.items():
                if month_str in dob_input:
                    month_from_name = month_num
                    month_name_found = month_str
                    break
            
            # Determine what information we have
            day = None
            month = None
            year = None
            missing = []
            
            # Case 1: Month name found with 2 numbers (day and year)
            if month_from_name and len(numbers) == 2:
                day = int(numbers[0])
                month = month_from_name
                year = int(numbers[1])
            # Case 2: Month name found with 1 number (could be day or year)
            elif month_from_name and len(numbers) == 1:
                num = int(numbers[0])
                month = month_from_name
                # If number is > 31, it's likely a year
                if num > 31:
                    year = num
                    missing.append("day")
                else:
                    day = num
                    missing.append("year")
            # Case 3: Only month name, no numbers
            elif month_from_name and len(numbers) == 0:
                month = month_from_name
                missing.extend(["day", "year"])
            # Case 4: 3 numbers (complete date)
            elif len(numbers) == 3:
                day = int(numbers[0])
                month = int(numbers[1])
                year = int(numbers[2])
            # Case 5: 2 numbers (missing one component)
            elif len(numbers) == 2:
                # Try to infer what's missing
                num1, num2 = int(numbers[0]), int(numbers[1])
                if num2 > 31:  # Second number is likely year
                    day = num1
                    year = num2
                    missing.append("month")
                else:
                    day = num1
                    month = num2
                    missing.append("year")
            # Case 6: 1 number (missing two components)
            elif len(numbers) == 1:
                num = int(numbers[0])
                if num > 31:  # Likely a year
                    year = num
                    missing.extend(["day", "month"])
                elif num > 12:  # Likely a day
                    day = num
                    missing.extend(["month", "year"])
                else:  # Could be day or month, assume day
                    day = num
                    missing.extend(["month", "year"])
            # Case 7: No numbers found
            else:
                missing = ["day", "month", "year"]
            
            # If anything is missing, return a helpful message
            if missing:
                missing_str = ', '.join(missing)
                return f"I found some information but need more details. Missing: {missing_str}. Please provide the complete date of birth with day, month, and year. Example: 15 August 2000 or 15/8/2000"
            
            # Validate the extracted values
            if not (1 <= day <= 31):
                return f"Invalid day: {day}. Day must be between 1 and 31."
            if not (1 <= month <= 12):
                return f"Invalid month: {month}. Month must be between 1 and 12."
            if year < 1900 or year > date.today().year:
                return f"Invalid year: {year}. Year must be between 1900 and {date.today().year}."
            
            # Calculate age
            try:
                birth_date = date(year, month, day)
                today = date.today()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                
                if age < 0:
                    return "Invalid date: The birth date is in the future."
                
                return f"The age is {age} years."
            except ValueError as e:
                return f"Invalid date combination. {str(e)}. Please check that the day is valid for the given month and year."
        
        elif isinstance(dob_input, (tuple, list)):
            if len(dob_input) < 3:
                missing = []
                if len(dob_input) < 1 or dob_input[0] is None:
                    missing.append("day")
                if len(dob_input) < 2 or dob_input[1] is None:
                    missing.append("month")
                if len(dob_input) < 3 or dob_input[2] is None:
                    missing.append("year")
                return f"Incomplete date. Missing: {', '.join(missing)}. Please provide all three values: day, month, year."
            
            dob_tuple = dob_input
            day, month, year = dob_tuple
            
            # Validate
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
        else:
            return "Please provide your date of birth. Example: 15 August 2000 or 15/8/2000"



class WeatherAgent:
    def __init__(self, api_key=None):
        """Initialize the WeatherAgent with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPEN_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.api_key)
        self.tools = self._initialize_tools()
        self.agent = None
    
    def _initialize_tools(self):
        """Initialize tools as class instances"""
        age_tool = AgeCalculatorTool()
        weather_tool = WeatherTool()
        return [age_tool, weather_tool]
    
    def _create_agent_with_memory(self, memory):
        """Create agent instance with specific memory"""
        PREFIX = """You are OsAMA, an intelligent assistant that helps users with their queries.

CRITICAL RULES FOR OUTPUT FORMAT:
- You MUST ALWAYS end your response with "Final Answer: [your response]"
- Even for simple greetings or conversations, use the format: "Final Answer: [your greeting/response]"
- This is REQUIRED for the system to work properly

YOUR CAPABILITIES:
- You can answer questions using your available tools OR respond to social/conversational messages
- You have TWO tools:
  1. Age Calculator - **MANDATORY** for ANY age/date of birth query
  2. Weather Forecast - for getting weather information for any location

*** CRITICAL AGE CALCULATOR RULES ***
- **YOU MUST CALL THE AGE CALCULATOR TOOL FOR ANY AGE-RELATED QUERY**
- This includes incomplete inputs: "July 1999", "born in 2002", "my birthday is August"
- **NEVER** answer age questions yourself - **ALWAYS** use the tool first
- Pass the user's input EXACTLY as they provided it to the Age Calculator tool
- The tool will detect missing information and ask the user for it
- Examples requiring tool usage:
  * "July 1999" → USE TOOL (not "please provide the day")
  * "I was born in 2002" → USE TOOL
  * "calculate my age" → USE TOOL
  * "my birthday is August 15" → USE TOOL
- Only after the tool responds can you relay its message to the user
  
WEATHER FORECAST RULES:
- Use this tool when users ask about weather, forecast, or climate conditions
- The tool automatically extracts location and number of days from the user's query
- Examples: "weather in London", "forecast for Paris 5 days", "what's the weather in Tokyo"
- If days are not mentioned, it provides current weather
  
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
- You have access to the full chat history with this user
- When users ask "what did I ask before" or "what questions did I ask from you" or similar, review the chat_history and provide accurate information
- Reference previous conversations naturally when relevant
- When asked about age modifications (like "add 5 to my age"), refer to the previously calculated age in chat history

LIMITATIONS:
- Do not answer general knowledge questions (e.g., "when was Pakistan created")
- The Document QA tool can ONLY answer questions about content in uploaded documents
- If Document QA returns "I cannot find information", tell the user this information is not in the documents
- For topics NOT related to age calculation, weather, or uploaded documents, politely inform the user about your capabilities

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
        """Process user message with manually provided chat history"""
        try:
            # Create memory and load chat history manually
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            
            # Load chat history if provided
            if chat_history:
                for exchange in chat_history:
                    if "user" in exchange and "AI" in exchange:
                        memory.chat_memory.add_user_message(exchange["user"])
                        memory.chat_memory.add_ai_message(exchange["AI"])
            
            # Create agent with memory
            agent = self._create_agent_with_memory(memory)
            response = agent.invoke({"input": user_message})
            
            return {"response": response["output"]}, 200
        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}, 500


weather_agent = WeatherAgent()


@weather_bp.route("/weather", methods=["POST"])
def weather_route():
    data = request.get_json()
    user_message = data.get("message")
    chat_history = data.get("chat_history", [])  

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    result = weather_agent.process_message(user_message, chat_history)
    return jsonify(result)