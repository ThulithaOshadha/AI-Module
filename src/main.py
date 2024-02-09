from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.ml import train_model

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    fuelType: str
    sex: int
    cp: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/train-model")
# async def train(inputData: InputData):
async def train():
    # model = train_model(inputData)
    model = train_model()
    return {"objects": model}


