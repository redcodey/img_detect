import os
import time
import sys
from typing import Union,Annotated
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File,Form
from fastapi.staticfiles import StaticFiles

from detect import lifespan,model_ai

#code version
__APP_NAME__ = "<Img Detect>"
__APP_VERSION__ = "0.0.1a"
__DATE_VERSION__ = 20240423
print(f'start {__APP_NAME__} v-{__APP_VERSION__}.{__DATE_VERSION__}')
print(f'time: {datetime.now()}')
print('--------------------------------------------------------------------')

app = FastAPI(lifespan=lifespan)

app.mount("/imgs", StaticFiles(directory="test_predictions"), name='images')

class Brand:
    brand_list = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/img2brand")
def img_2_brand(file: UploadFile = File(...)):
    dectectAPI =model_ai["DectectAPI"]
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
        
    dt = dectectAPI.DectImg2Brand(file_location)
    print(f"detect: {dt}")

    return dt
    
@app.post("/detect-brand")
def detect_brand(file: UploadFile = File(...) ):
    dectectAPI =model_ai["DectectAPI"]
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
        
    dt = dectectAPI.DectBrand(file_location)
    print(f"detect-brand: {dt}")

    return dt

# detect brands from img url
@app.post("/imgurl2brand")
def img_2_brand(image_url: Annotated[str, Form()], image_name:Annotated[str, Form()], project_id: Annotated[str, Form()]):
    dectectAPI =model_ai["DectectAPI"]
    #print('dt')    
    dt = dectectAPI.DectImgUrl2Brand(image_url, image_name)
    #print(f"detect: {dt}")

    return dt