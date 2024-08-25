Setup and Configuration

1) Clone this github repository 
    ```
    git clone https://github.com/YASHkh2609/Voice_assistant.git
    ```

2) Environment Setup:

    Python: Ensure Python 3.8 or higher is installed.

    Virtual Environment: Create a virtual environment to manage dependencies

    Interpreter: Select the corresponding python interpretor (ctrl+shift+P - to open command palette and select interpretor)

      ```
   python -m venv venv
   venv\Scripts\activate
      ```

4) Library Installation:
`pip install -r requirements.txt`

5) API Keys and Credentials:
   
    Deepgram API: Sign up for a Deepgram account and obtain API key.
   
    Groq: Register and get access to Groq’s API key for the llama3-70b-8192 model.
   
    Database Access: Set up access credentials for SQL or MongoDB databases as needed.


6) Create a configuration file (config.json) to store database connection details.

```
{
  "database":
      {
        "type": "mysql",
        "host": "localhost",
        "user": "root",
        "password": "password",
        "database": "product_db"
       }
}
```

7) Create a .env to store API keys
```
    deepgram_api_key: "YOUR_DEEPGRAM_API_KEY",
    groq_api_key": "YOUR_GROQ_API_KEY"
```

8) Run main.py to interact with the voice assistant using the terminal
` python main.py`
