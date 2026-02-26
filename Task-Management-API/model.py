from pydantic import BaseModel,Field
from typing import Annotated,Optional,Literal

class CreateTask(BaseModel):
     title:Annotated[str,Field(...,description="Title of the task")]
     description:Optional[str]=None
     priority:Annotated[Literal["high","medium","low"],Field(...,description="What is the priority of your task")]
     
