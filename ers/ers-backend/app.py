from flask import Flask, request, jsonify
from utils.model_service import biomedical_ner

app = Flask(__name__)
app.config['MongoDB_Connection_String'] = ''


@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

@app.route('/biomedical_ner', methods=['POST'])


def model_use():
    """
    Performs biomedical NER on the provided text using the loaded model.

    Expects a JSON request with the following structure:
    ```json
    {
        "text": "Your text here"
    }
    ```

    Returns a JSON response with the processed text and filtered entities.
    """
    # logging.info(f"Request: {request}")  # Log the request object for debugging

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get("text", 'None')

    if not text:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    # Perform NER and filter entities
    results = biomedical_ner(text)
    filtered_entities = [entity for entity in results if entity['score'] > 0.1]  # Confidence threshold

    # Ensure all entity scores are cast to float (JSON can't handle numpy.float32)
    for entity in filtered_entities:
        entity['score'] = float(entity['score'])  # Convert score to standard float

    response = {
        "text": text,
        "entities": filtered_entities
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)