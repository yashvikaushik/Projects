from flask import Flask, request, jsonify
import random
import re
import json

app = Flask(__name__, static_folder='static')

def load_bank():
    try:
        with open('questions_data.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print("Failed to load questions_data.json", e)
        return {}

# Load expanded Question Bank of 200 entries dynamically
QUESTION_BANK = load_bank()

def get_question_by_id(qid):
    for domain, questions in QUESTION_BANK.items():
        for q in questions:
            if q["id"] == qid:
                return q
    return None

def analyze_answer(answer_text, question):
    text = answer_text.lower()
    words = set(re.findall(r'\b\w+\b', text))
    
    matched_keywords = []
    missing_keywords = []
    for kw in question["keywords"]:
        if kw.lower() in text:
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)
            
    coverage = len(matched_keywords) / len(question["keywords"]) if question["keywords"] else 1
    
    word_count = len(words)
    penalty = 0
    feedback = []
    
    if word_count < 8:
        penalty = 30
        feedback.append("Too brief. Elaborate on the specifics.")
    elif word_count > 120:
        penalty = 15
        feedback.append("Highly verbose. Be more concise.")
        
    score = max(0, min(100, int((coverage * 100) - penalty)))
    
    if coverage < 0.4:
        feedback.append("Missing core technical concepts. Study this topic.")
    elif score > 75:
        feedback.insert(0, "Great answer! You hit the main points.")
        
    if not feedback:
        feedback.append("Good structure, but include more domain keywords.")
        
    return {
        "score": score,
        "missing_concepts": missing_keywords[:4],
        "feedback": feedback,
        "ideal_answer": question["ideal_answer"]
    }

@app.route('/api/domains', methods=['GET'])
def get_domains():
    domains = [{"id": k, "name": k, "count": len(v)} for k, v in QUESTION_BANK.items()]
    return jsonify(domains)

@app.route('/api/questions/<domain>', methods=['GET'])
def get_domain_questions(domain):
    if domain not in QUESTION_BANK:
        return jsonify({"error": "Domain not found"}), 404
        
    questions = list(QUESTION_BANK[domain])
    random.shuffle(questions)
    
    safe_q = [{"id": q["id"], "question": q["question"]} for q in questions]
    return jsonify(safe_q)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    q_id = data.get("question_id")
    answer = data.get("answer", "")
    
    if not answer.strip():
        return jsonify({"error": "Answer cannot be empty"}), 400
        
    question = get_question_by_id(q_id)
    if not question:
        return jsonify({"error": "Question not found"}), 404
        
    result = analyze_answer(answer, question)
    return jsonify(result)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
