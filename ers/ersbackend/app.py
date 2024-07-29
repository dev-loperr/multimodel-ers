from flask import Flask, request, jsonify
# from "multimodel-ers".ers.ersbackend.utils.model_app import process_text
from utils.model_app import process_text

app = Flask(__name__)
app.config['MongoDB_Connection_String'] = ''

@app.route("/process_text", methods=['POST', 'GET'])
def handle_text():
    text = request.get_json().get("text")
    entities = process_text(text)
    return {"ans": entities}

@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

if __name__ == "__main__":
    app.run(debug=True)