from fastapi import FastAPI,Path,HTTPException,Query
import json
from fastapi.responses import JSONResponse
from typing import Annotated,Literal,Optional
from pydantic import BaseModel,Field,computed_field
from fastapi.middleware.cors import CORSMiddleware


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, PUT, DELETE
    allow_headers=["*"],
)

class Patient(BaseModel):
    id :  Annotated[str,Field(...,description="ID OF THE PATIENT",examples=['P001'])]
    name: Annotated[str,Field(...,description="Name of the patient")]
    city: Annotated[str,Field(description="City of the patient")]
    age: Annotated[int,Field(...,gt=0,le=120,description="Age of the patient")]
    gender:Annotated[Literal['male','female','other'],Field(...,description="Gender of the patient")]
    height:Annotated[float,Field(...,gt=0,description='height of the patient')]
    weight: Annotated[float,Field(...,gt=0,description='weight of the patient')]
    
    @computed_field
    @property
    def bmi(self) -> float:
        bmi=round(self.weight/self.height**2,2)
        return bmi
    
    @computed_field
    @property
    def verdict(self)->str:
        if self.bmi<18.5:
            return 'underweight'
        elif self.bmi<25:
            return 'normal'
        elif self.bmi<30:
            return 'normal'
        else :
            return 'obese'
        
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str],Field(default=None)]
    city: Annotated[Optional[str],Field(default=None)]
    age: Annotated[Optional[int],Field(default=None,gt=0)]
    gender:Annotated[Optional[Literal['male','female','other']],Field(default=None)]
    height:Annotated[Optional[float],Field(default=None,gt=0)]
    weight: Annotated[Optional[float],Field(default=None,gt=0)]

        



def load_data():
    with open('patients.json','r') as f:
        data=json.load(f)
    return data
    

@app.get("/")
def hello():
    return {"message: Patient management Syatem "}

@app.get("/about")
def about():
    return{"message: a fully functional API to manage your patient records"}

@app.get("/view")
def view():
    data=load_data()
    return data

@app.get("/patient/{patient_id}")
def view_patient(patient_id:str=Path(...,description="Description of the patient in DB",example="P00N")):
    data=load_data()
    if patient_id in data:
        return data[patient_id]
    else:
        raise HTTPException(status_code=404,detail='invalid patient ID ')
    
@app.get("/sort")
def sort_patients(sort_by: str=Query(...,description='sort on the basis of heigth weigth or bmi'),order: str=Query('asc',description="sort in ascending or descending order")):
    valid_fields=['height','weight','bmi']

    if sort_by not in valid_fields:
        raise HTTPException(status_code=400,detail='invalid field')
    
    
    if order not in ['asc','desc']:
        raise HTTPException(status_code=400,detail='invalid order')
    
    data=load_data()

    sort_order=True if order=='desc' else False

    sorted_data=sorted(data.values(),key=lambda x: x.get(sort_by,0),reverse=sort_order)

    return sorted_data

def save_data(data):
    with open('patients.json','w') as f:
        json.dump(data,f,indent=4)

@app.post("/create")
def  create_patient(patient:Patient):
    data=load_data()
    if patient.id in data:
        raise HTTPException(status_code=400,detail='patient exists already')

    patient_dict = patient.model_dump()
    patient_id = patient_dict.pop("id")

    data[patient_id] = patient_dict

    save_data(data)

    return JSONResponse(status_code=200,content={'message':'patient created successfully'})


@app.put("/edit/{patient_id}")
def update_patient(patient_id:str,patient_update:PatientUpdate):
    data=load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404,detail='patient not found')
    
    existing_patient_info=data[patient_id]

    updated_patient_info= patient_update.model_dump(exclude_unset=True)

    for key,value in updated_patient_info.items():
        existing_patient_info[key]=value
    
    existing_patient_info['id']=patient_id
    patient_pydantic_object=Patient(**existing_patient_info)

    existing_patient_info=patient_pydantic_object.model_dump(exclude={'id'})

    data[patient_id]=existing_patient_info

    save_data(data)

    return JSONResponse(status_code=200,content={'message':'patient updated'})


@app.delete('/delete/{patient_id}')
def delete_patient(patient_id:str):
    data=load_data()
    if patient_id not in data:
        raise HTTPException(status_code=400,detail="patient not found")
    
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200,content={'message':'deleted'})






        


