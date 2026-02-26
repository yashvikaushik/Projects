import json
from model import CreateTask 
from datetime import datetime
from uuid import uuid4

FILE_NAME="tasks.json"

def read_data():
    with open(FILE_NAME,'r') as f:
        return json.load(f)
    
def write_data(tasks):
    with open(FILE_NAME,'w') as f:
        json.dump(tasks, f, indent=4)

def create_task(task:CreateTask):
    tasks=read_data()

    new_task={
        "id": str(uuid4()),
        "title": task.title,
        "description": task.description,
        "priority": task.priority,
        "completed": False,
        "created_at": datetime.utcnow().isoformat()
    }

    tasks.append(new_task)
    write_data(tasks)

    return new_task

def show_tasks():
    tasks=read_data()
    return tasks