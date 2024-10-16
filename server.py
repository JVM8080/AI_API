import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()


model = tf.keras.models.load_model('app/model.h5')

class ImageVector(BaseModel):
    image_vector: list  

@app.get("/prueba")
def read_root():
    return {"message": "Esta es una prueba hola"}

@app.post("/predict")
async def predict(image_data: ImageVector):
    try:
        
        img = np.array(image_data.image_vector, dtype=float).reshape(-1, 224, 224, 3)
        prediction = model.predict(img)
        category_index = np.argmax(prediction[0], axis=-1)
        category_name = switch_case(category_index)

        return {"category": category_name}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



def switch_case(value):
    if value == 0:
        return "Banana"
    elif value == 1:
        return "Botella"
    elif value == 2:
        return "Manzana"
    elif value == 3:
        return "Papel"
    else:
        return "Opción no válida"
