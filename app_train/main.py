import os
import time
import sys
from typing import Union
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File

import detect as DectectAPI

#code version
__APP_NAME__ = "<Img Detect>"
__APP_VERSION__ = "0.0.1a"
__DATE_VERSION__ = 20240423
print(f'start {__APP_NAME__} v-{__APP_VERSION__}.{__DATE_VERSION__}')
print(f'time: {datetime.now()}')
print('--------------------------------------------------------------------')

app = FastAPI()

class Brand:
    brand_list = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/img2brand")
def img_2_brand(file: UploadFile = File(...), project_id: str = "30"):
    file_location = f"files/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        
        
    dt = DectectAPI.DectImg2Brand(file_location)
    print(f"detect: {dt}")
    
    brands = Brand()
    brands.brand_list = dt
    
    return brands
    