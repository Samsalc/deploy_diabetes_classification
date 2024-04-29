from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib


app = FastAPI(
    title="Deploy diabetes classification model",
    version="0.0.1"
)

# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
model = joblib.load("model/logistic_regression_model_v1.pkl")


@app.post("/api/v1/diabetes-classification", tags=["diabetes"])
async def predict(
    Age: float,
    Gender: float,
    BMI: float,
    Chol: float,
    TG: float,
    HDL: float,
    LDL: float,
    Cr: float,
    BUN: float
):

    dictionary = {
        'Age': Age,
        'Gender': Gender,
        'BMI': BMI,
        'Chol': Chol,
        'TG': TG,
        'HDL': HDL,
        'LDL': LDL,
        'Cr': Cr,
        'BUN': BUN
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content="N"
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

