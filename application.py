from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

application = Flask(__name__)
CORS(application)


from routes.home_route import home_bp
from routes.create_embedding import embedding_bp
from routes.query_route import query_bp
from routes.agent_route import agent_bp


application.register_blueprint(home_bp)
application.register_blueprint(embedding_bp)
application.register_blueprint(query_bp)
application.register_blueprint(agent_bp)

if __name__ == "__main__":
    application.run(debug=True)