from flask import Flask, request, jsonify
from model_service import biomedical_ner
# import logging

text="""leave ventricular dysfunction propofol use uncommon complication require increase awareness 
 peri operative management propofol widely anesthetic agent generally predictable 
 adverse effect profile propose patient pre existing left ventricular dysfunction propofol 
 judiciously standardized management protocol lv dysfunction arise paucity 
 exist literature propofol effect myocardial contractility randomize control trial 
 necessary elucidate datum uncommon effect 
 additional information 
 disclosure 
 human subject consent obtain waive participant study conflict interest 
 compliance icmje uniform disclosure form author declare following payment service 
 info author declare financial support receive organization 
 submit work financial relationship author declare financial 
 relationship present previous year organization 
 interest submit work relationship author declare 
 relationship activity appear influence submit work 
 reference 
 1 
 hug cc jr mcleskey ch nahrwold ml et al hemodynamic effect propofol datum 25,000 
 patient anesth analg 1993 77:21 29 
 2 
 garimella b elnadoury o khorolsky c iskandir c mercado j effect propofol vasopressor 
 requirement base underlie cardiac status patient vasodilatory shock chest 2019 
 156 a1040 10.1016 j.chest.2019.08.960 
 3 
 sprung j ogletree hughes ml mcconnell bk zakhary dr smolsky sm moravec cs effect 
 propofol contractility fai"""
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World! Biomedical NER server is up and running.'

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

if __name__ == '__main__':
    app.run(debug=True, port=6000)





# from flask import Flask, request, jsonify
# from model_service import biomedical_ner

# app = Flask(__name__)
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'



# @app.route('/biomedical_ner', methods=['GET'])
# def model_use():

#     text1 = """
#  dlss pv drugsafety 
 
#  3 
#  sprung j ogletree hughes ml mcconnell bk zakhary dr smolsky sm moravec cs effect 
#  propofol contractility fail nonfaile human heart muscle anesth analg 2001 93:550 9 
#  10.1097/00000539 200109000 00006 
#  4 
#  divanji p nah g harris agarwal parikh ni risk factor recurrent peripartum cardiomyopathy 
#  783 woman california circulation 2018 a17220:138 10.1161 circ.138.suppl_1.17220 
#  5 
#  kassam si lu c buckley n lee rm mechanism propofol induce vascular relaxation 
#  modulation perivascular adipose tissue endothelium anesth analg 2011 112:1339 45 
#  10.1213 ane.0b013e318215e094 
#  2023 karan et al cureus 15(7 e41815 doi 10.7759 cureus.41815 
#  3 3 


# """

#     ans1 = biomedical_ner(text1)
#     return {
#         "ans": ans1
#     }

    

# if __name__ == "__main__":
#     app.run(debug=True)










# from flask import Flask, request, jsonify
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# # Load the pre-trained biomedical entity recognition model (assuming it's available)
# model_checkpoint = 'jsylee/scibert_scivocab_uncased-finetuned-ner'
# model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
#                                                        num_labels=5,
#                                                        id2label={0: 'O', 1: 'DRUG', 2: 'PROTEIN', 3: 'ADVERSE EFFECT', 4: 'CELL_TYPE'})
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
# tokenizer.model_max_length = 512

# model_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     """Returns a simple greeting message."""
#     return 'Hello, World! Biomedical NER server is up and running.'

# @app.route('/biomedical_ner', methods=['POST'])
# def model_use():
#     """Performs biomedical NER on the provided text using the loaded model.

#     Expects a JSON request with the following structure:
#     ```json
#     {
#         "text": "Your text here"
#     }
#     ```

#     Returns a JSON response with the processed text and filtered entities.
#     """

#     if not request.is_json:
#         return jsonify({"error": "Request must be JSON"}), 400

#     data = request.get_json()
#     text = data.get("text", None)

#     if not text:
#         return jsonify({"error": "Missing 'text' field in request body"}), 400

#     # Perform NER and filter entities
#     results = model_pipeline(text)
#     filtered_entities = [entity for entity in results if entity['score'] > 0.1]

#     response = {
#         "text": text,
#         "entities": filtered_entities
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True, port=6000)

#create endpoint to send request to model on server
# from flask import Flask, request, jsonify
# from model_service import biomedical_ner
# text1 = """
#   dlss pv drugsafety 
 
#   3 
#   sprung j ogletree hughes ml mcconnell bk zakhary dr smolsky sm moravec cs effect 
#   propofol contractility fail nonfaile human heart muscle anesth analg 2001 93:550 9 
#   10.1097/00000539 200109000 00006 
#   4 
#   divanji p nah g harris agarwal parikh ni risk factor recurrent peripartum cardiomyopathy 
#   783 woman california circulation 2018 a17220:138 10.1161 circ.138.suppl_1.17220 
#   5 
#   kassam si lu c buckley n lee rm mechanism propofol induce vascular relaxation 
#   modulation perivascular adipose tissue endothelium anesth analg 2011 112:1339 45 
#   10.1213 ane.0b013e318215e094 
#   2023 karan et al cureus 15(7 e41815 doi 10.7759 cureus.41815 
#   3 3 


#  """

# app = Flask(__name__)
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'
# @app.route('/biomedical_ner', methods=['GET'])
# def model_use():
#     text1 = request.args.get('text')
#     ans1 = biomedical_ner(text1)
#     return {
#         "ans": ans1
#     }

# if __name__ == "__main__":
#      app.run(debug=True, port = 6000)
# #debug the error occuring in above code
