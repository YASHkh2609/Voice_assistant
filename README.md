Setup and Configuration

1) Clone this github repository

2) Environment Setup:
Python: Ensure Python 3.8 or higher is installed.
Virtual Environment: Create a virtual environment to manage dependencies
Interpreter: Select the corresponding python interpretor (ctrl+shift+P - to open command palette and select interpretor)

'''python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`'''

3) Library Installation:
'''pip install -r requirements.txt'''

4) API Keys and Credentials:
Deepgram API: Sign up for a Deepgram account and obtain API key.
Groq: Register and get access to Groq’s API key for the llama3-70b-8192 model.
Database Access: Set up access credentials for SQL or MongoDB databases as needed.

5) Create a configuration file (config.json) to store database connection details.
''' 
{
  "database": {
    "type": "mysql",
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "product_db"
  }
}
'''
6) Create a .env to store API keys
''' deepgram_api_key: "YOUR_DEEPGRAM_API_KEY",
    groq_api_key": "YOUR_GROQ_API_KEY"
'''
7) Run main.py to interact with the voice assistant using the terminal