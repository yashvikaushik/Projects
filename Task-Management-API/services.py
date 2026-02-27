import json
from model import CreateTask,UpdateTask
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
    tasks=read_data() #is a python list

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


def update_task(task_id:str,update_data:UpdateTask):
    tasks=read_data()

    for task in tasks:
         if task["id"] == task_id:
             if update_data.description is not None:
                task["description"] = update_data.description

             if update_data.completed is not None:
                task["completed"] = update_data.completed
        
         write_data(tasks)
         return task
    return None
    

def delete_task(task_id:str):
    tasks=read_data()
    for task in tasks:
        if task["id"]==task_id:
            if task["completed"] == True:
                tasks.remove(task)
                write_data(tasks)
                return {"message": "Task deleted successfully"}
            else:
                return {"message": "Task is not completed yet"}

    return {"error": "Task not found"}
