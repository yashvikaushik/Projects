import json
import uuid

base_data = {
  "OOP": [
    {"question": "What is Object-Oriented Programming?", "keywords": ["encapsulation", "inheritance", "polymorphism", "abstraction", "paradigm"]},
    {"question": "What is encapsulation?", "keywords": ["data hiding", "class", "access", "private", "bundle"]},
    {"question": "What is inheritance?", "keywords": ["reuse", "parent", "child", "extend", "base", "derived"]},
    {"question": "What is polymorphism?", "keywords": ["overloading", "overriding", "many forms", "same interface"]},
    {"question": "What is abstraction?", "keywords": ["hide", "details", "interface", "essential", "background"]},
    {"question": "Difference between class and object?", "keywords": ["blueprint", "instance", "template", "memory"]},
    {"question": "What is a constructor?", "keywords": ["initialize", "object", "instantiation", "memory"]},
    {"question": "What is method overloading?", "keywords": ["same name", "different parameters", "compile time", "static"]},
    {"question": "What is method overriding?", "keywords": ["runtime", "inheritance", "dynamic", "same signature"]},
    {"question": "What is an access modifier?", "keywords": ["public", "private", "protected", "visibility", "scope"]},
  ],
  "DBMS": [
    {"question": "What is DBMS?", "keywords": ["database", "management", "system", "store", "retrieve"]},
    {"question": "What is normalization?", "keywords": ["redundancy", "normal forms", "anomalies", "organize"]},
    {"question": "What is a primary key?", "keywords": ["unique", "identifier", "not null", "table"]},
    {"question": "What is a foreign key?", "keywords": ["reference", "relation", "link", "parent table"]},
    {"question": "What is SQL?", "keywords": ["query", "database", "structured", "manipulate"]},
    {"question": "Difference between SQL and NoSQL?", "keywords": ["structured", "unstructured", "relational", "document", "schema"]},
    {"question": "What is indexing?", "keywords": ["search", "optimize", "speed", "data structure", "b-tree"]},
    {"question": "What is a transaction?", "keywords": ["ACID", "atomicity", "unit of work", "commit", "rollback"]},
    {"question": "What is the ACID property?", "keywords": ["atomicity", "consistency", "isolation", "durability", "reliable"]},
    {"question": "What is a join?", "keywords": ["combine", "tables", "records", "foreign key", "inner", "outer"]}
  ],
  "OS": [
    {"question": "What is an operating system?", "keywords": ["interface", "hardware", "software", "manage", "resources"]},
    {"question": "What is a process?", "keywords": ["execution", "program", "memory", "pcb", "active"]},
    {"question": "What is a thread?", "keywords": ["lightweight", "process", "execution unit", "shared memory"]},
    {"question": "What is a deadlock?", "keywords": ["waiting", "resources", "circular", "block", "hold and wait"]},
    {"question": "What is CPU scheduling?", "keywords": ["CPU", "process", "time", "algorithm", "fairness", "throughput"]},
    {"question": "Difference between process and thread?", "keywords": ["memory", "execution", "shared", "isolated", "context switch"]},
    {"question": "What is paging?", "keywords": ["memory", "pages", "non-contiguous", "frames", "physical"]},
    {"question": "What is segmentation?", "keywords": ["memory", "segments", "logical", "variable size", "user view"]},
    {"question": "What is virtual memory?", "keywords": ["disk", "RAM", "illusion", "large", "swap"]},
    {"question": "What is context switching?", "keywords": ["CPU", "switch", "save state", "load state", "overhead"]}
  ],
  "HR": [
    {"question": "Tell me about yourself", "keywords": ["background", "skills", "goals", "experience", "journey"]},
    {"question": "Why should we hire you?", "keywords": ["skills", "value", "fit", "contribute", "asset"]},
    {"question": "What are your greatest strengths?", "keywords": ["skills", "qualities", "dedicated", "problem solving", "adaptable"]},
    {"question": "What are your weaknesses?", "keywords": ["improvement", "learning", "overcoming", "aware", "steps"]},
    {"question": "Where do you see yourself in 5 years?", "keywords": ["growth", "career", "leadership", "impact", "senior"]},
    {"question": "Why do you want this job?", "keywords": ["interest", "company", "culture", "mission", "align"]},
    {"question": "Describe a challenge you faced and how you handled it", "keywords": ["problem", "solution", "action", "result", "STAR"]},
    {"question": "How do you handle stress and pressure?", "keywords": ["manage", "pressure", "prioritize", "calm", "focus"]},
    {"question": "Are you a team player?", "keywords": ["teamwork", "collaboration", "support", "goal", "communicate"]},
    {"question": "Why did you leave your last job?", "keywords": ["growth", "reason", "opportunity", "advance", "challenge"]}
  ],
  "ML": [
    {"question": "What is machine learning?", "keywords": ["data", "model", "training", "learn", "predict"]},
    {"question": "What is supervised learning?", "keywords": ["labeled", "data", "target", "predict", "examples"]},
    {"question": "What is unsupervised learning?", "keywords": ["unlabeled", "data", "hidden", "structure", "patterns", "clustering"]},
    {"question": "What is overfitting?", "keywords": ["training", "generalization", "noise", "high variance", "memorize"]},
    {"question": "What is underfitting?", "keywords": ["model", "simple", "high bias", "poor performance", "capture"]},
    {"question": "What is regression?", "keywords": ["continuous", "prediction", "output", "value", "linear"]},
    {"question": "What is classification?", "keywords": ["categories", "labels", "discrete", "classes", "predict"]},
    {"question": "What is KNN?", "keywords": ["distance", "neighbors", "majority", "lazy", "algorithm"]},
    {"question": "What is a decision tree?", "keywords": ["nodes", "splitting", "rules", "leaves", "branches", "entropy"]},
    {"question": "Explain Bias vs Variance tradeoff", "keywords": ["error", "tradeoff", "underfitting", "overfitting", "balance", "complexity"]}
  ]
}

templates = [
    ("{orig_q}", "The fundamental answer should revolve around {keys}."),
    ("Explain the concept of {topic} in detail.", "Focus on explaining {topic} thoroughly. Key concepts include: {keys}."),
    ("How would you describe {topic} to a beginner?", "Use simple terms to describe {topic}. Mention aspects like {keys}."),
    ("What are the core advantages or characteristics of {topic}?", "Discuss the benefits and traits of {topic}, specifically referencing {keys}.")
]

expanded_bank = {}

for domain, questions in base_data.items():
    expanded_bank[domain] = []
    
    for q in questions:
        # derive a base topic from the question by stripping common prefixes
        base_topic = q["question"]\
            .replace("What is ", "")\
            .replace("What are ", "")\
            .replace("?", "")\
            .replace("Difference between ", "")\
            .replace("Explain ", "")\
            .replace("Tell me about ", "")\
            .replace("Why should we ", "")\
            .replace("Where do you ", "")\
            .replace("Why do you ", "")\
            .replace("How do you ", "")
            
        
        for i, (t_q, t_ans) in enumerate(templates):
            if "{orig_q}" in t_q:
                new_q = q["question"]
            else:
                new_q = t_q.format(topic=base_topic)
                
            keys_str = ", ".join(q["keywords"][:3])
            new_ans = t_ans.format(topic=base_topic, keys=keys_str)
            
            # format capitalization properly
            new_q = new_q[0].upper() + new_q[1:] if len(new_q) > 0 else new_q
            
            expanded_bank[domain].append({
                "id": str(uuid.uuid4())[:8],
                "question": new_q,
                "keywords": q["keywords"],
                "ideal_answer": new_ans
            })

with open('questions_data.json', 'w') as f:
    json.dump(expanded_bank, f, indent=4)

print("Successfully generated 200 questions into questions_data.json")
