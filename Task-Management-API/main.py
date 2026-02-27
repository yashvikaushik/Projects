from fastapi import FastAPI
from services import create_task,show_tasks,update_task,delete_task
from model import CreateTask,UpdateTask


app=FastAPI()

@app.get('/')
def home_page():
    return {"message":"this is your home page"}

@app.post("/create_task")
def create_new_task(task:CreateTask):
    return create_task(task)
         
@app.get("/show_tasks")
def show_all_tasks():
      return show_tasks()

@app.put("/update_task/{task_id}")
def update_old_task(task_id:str,update_tasks:UpdateTask):
     return update_task(task_id,update_tasks)

@app.delete("/delete_task/{task_id}")
def delete_completed_task(task_id:str):
     return delete_task(task_id)
