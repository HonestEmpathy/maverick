import os
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

def infer_severity_from_context(text, logits):
    severity_reasons = []
    probs = softmax(logits, dim=1).tolist()[0]

    if "asymptomatic" in text.lower() or "no symptoms" in text.lower():
        severity_level = 1
        severity_reasons.append("Condition is asymptomatic or minor.")
    elif "severe" in text.lower() or "life-threatening" in text.lower():
        severity_level = 4
        severity_reasons.append("Severe or life-threatening situation.")
    elif "moderate" in text.lower() or "requires help" in text.lower():
        severity_level = 2
        severity_reasons.append("Moderate complications, requiring assistance.")
    elif "dependency" in text.lower() or "constant care" in text.lower():
        severity_level = 3
        severity_reasons.append("High dependency, major intervention required.")
    else:
        severity_level = torch.argmax(logits).item() + 1
        severity_reasons.append(f"Model inferred severity from context: Level {severity_level}")
    return severity_level, severity_reasons

def map_group_from_severity(severity_level):
    if severity_level == 1:
        return "General Counselor"
    elif severity_level == 2:
        return "Trauma Specialist (Entry-Level) or Substance Abuse Specialist (Intermediate)"
    elif severity_level == 3:
        return "Trauma Specialist (Experienced) or Substance Abuse Specialist (Veteran)"
    elif severity_level == 4:
        return "Suicide Prevention (Advanced) or Trauma Specialist (Advanced)"
    else:
        return "Unknown Group"

def classify_severity_with_reasoning(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    severity_level, severity_reasons = infer_severity_from_context(input_text, logits)
    group = map_group_from_severity(severity_level)

    response = f"Severity: {severity_level}\nGroup: {group}\nReasoning: {', '.join(severity_reasons)}"
    return response

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    input_text = data.get('text', '')
    if input_text:
        result = classify_severity_with_reasoning(input_text)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    app.run(host='0.0.0.0', port=port)