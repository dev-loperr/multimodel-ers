from flask import Flask, request, jsonify
from utils.model1_app import chemical_disease_text
from utils.model2_app import jsylee
from utils.model3_app import extractor
from utils.model4_app import gliner
from utils.model5_app import fran_martinez
from utils.model6_app import gpt2
from utils.model7_app import biobert
from utils.model8_app import bert_base
import numpy as np
import logging

app = Flask(__name__)
app.config['MongoDB_Connection_String'] = ''

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred"}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(400)
def handle_400(e):
    logging.info(f"Bad request: {str(e)}")
    return jsonify({"error": "Bad request"}), 400


@app.route("/chemical_disease", methods=['POST', 'GET'])
def chemical_disease_endpoint():
    try:
        text = request.get_json().get("text")
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = chemical_disease_text(text)
        return jsonify(entities)
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model chemical_disease has run")

@app.route('/jsylee', methods=['POST'])
def jsylee_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = jsylee(text)
        native_entities = convert_to_native(entities)
        return jsonify(native_entities)
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model jsylee has run")

@app.route('/keyphrases', methods=['POST'])
def keyphrases_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        keyphrases = extractor.keyphrases(text)
        return jsonify(convert_to_native(keyphrases))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error extracting keyphrases: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model keyphrases has run")

@app.route('/gliner', methods=['POST'])
def gliner_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = gliner(text)
        return jsonify(convert_to_native(entities))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error extracting terms: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model gliner has run")

@app.route('/fran_martinez', methods=['POST'])
def fran_martinez_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = fran_martinez(text)
        return jsonify(convert_to_native(entities))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error extracting entities: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model fran-martinez has run")

@app.route('/gpt2', methods=['POST'])
def gpt2_endpoint():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        if not prompt:
            raise ValueError("Invalid input: 'prompt' field is required")
        generated_texts = gpt2(prompt)
        return jsonify(convert_to_native(generated_texts))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error generating text: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model gpt2 has run")

@app.route('/biobert', methods=['POST'])
def biobert_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = biobert(text)
        return jsonify(convert_to_native(entities))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error extracting BioBERT entities: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model biobert has run") 

@app.route('/bert_base', methods=['POST'])
def bert_base_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            raise ValueError("Invalid input: 'text' field is required")
        entities = bert_base(text)
        return jsonify(convert_to_native(entities))
    except ValueError as e:
        logging.info(f"Bad request: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Error extracting medical entities: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        logging.info("Model bert-base has run") 

@app.route('/', methods=['POST', 'GET'])
def index():
    return "ERS-Backend is up!"

if __name__ == "__main__":
    app.run(debug=True)
