from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Assuming you have installed the required libraries (transformers)
# You can install them using: pip install transformers

def biomedical_ner(text):

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
    # Load the pre-trained biomedical entity recognition model
    model_checkpoint = 'jsylee/scibert_scivocab_uncased-finetuned-ner'
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                           num_labels=5,
                                                           id2label={0: 'O', 1: 'DRUG', 2: 'PROTEIN',
                                                                    3: 'ADVERSE EFFECT', 4: 'CELL_TYPE'})
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer.model_max_length = 512

    model_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Use the loaded pipeline to perform NER
    result = model_pipeline(text)

    # No filtering or score threshold applied here (optional in model_use)
    return result  # Return all results for potential customization in model_use












# # jsylee/scibert_scivocab_uncased-finetuned-ner
# # from tokenizers import Tokenizer
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
# text1 = """
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
# import numpy as np
# def convert_to_native(obj):
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj

# # Assuming you have installed the required libraries (transformers)
# # You can install them using: pip install transformers



# def biomedical_ner(text1):
#     # Load the pre-trained biomedical entity recognition model
#     model_checkpoint = 'jsylee/scibert_scivocab_uncased-finetuned-ner'
#     model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
#                                                     num_labels=5,
#                                                     id2label={0: 'O', 1: 'DRUG', 2: 'PROTEIN', 3: 'ADVERSE EFFECT', 4: 'CELL_TYPE'}
#                                                     )
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
#     tokenizer.model_max_length = 512

#     model_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

#     # print((request))
#     # if not request.is_json:
#     #     return jsonify({"error": "Request must be JSON"}), 400

#     # data = request.get_json()
#     # text = data.get("text", None)

#     # if not text:
#     #     return jsonify({"error": "Missing 'text' field in request body"}), 400

#     # Use the loaded pipeline to perform NER
#     results = model_pipeline(text1)

#     # Filter entities with scores above 0.1 (confidence threshold)
#     filtered_entities = [entity for entity in results if entity['score'] > 0.1]

#     response = {
#         "text": text1,
#         "entities": filtered_entities
#     }
#     print(response)

#     return (response)

# # print(biomedical_ner(text1))


# # lines = file_text.splitlines()
# # for line in lines:
# #     result = model_pipeline(line)
# #     for entity in result:
# #         if (entity["score"] > 0.8):
# #             print(entity)
# #         if (entity['entity_group'] == 'ADVERSE EFFECT' and entity["score"] > 0.8):
# #             print(entity)
# #         if (entity['entity_group'] == 'protein' and entity["score"] > 0.8):
# #             print(entity)
# #         if (entity['entity_group'] == 'cell_Type' and entity["score"] > 0.8):
# #             print(entity)
