from fastapi import FastAPI
from services import create_task,show_tasks
from model import CreateTask


app=FastAPI()


@app.get('/')
def home_page():
    return {"message":"this is your home page"}

@app.post("/create-task")
def create_new_task(task:CreateTask):
    return create_task(task)
         
@app.get("/show-tasks")
def show_all_tasks():
      return show_tasks()