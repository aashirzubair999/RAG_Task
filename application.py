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



application.register_blueprint(home_bp)
application.register_blueprint(embedding_bp)

if __name__ == "__main__":
    application.run(debug=True)