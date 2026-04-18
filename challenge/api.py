import fastapi
import pandas as pd
from pydantic import BaseModel
from typing import List

from challenge.model import DelayModel


VALID_OPERA = {
    "Aerolineas Argentinas", "Aeromexico", "Air Canada", "Air France",
    "Alitalia", "American Airlines", "Austral", "Avianca", "British Airways",
    "Copa Air", "Delta Air", "Gol Trans", "Grupo LATAM", "Iberia",
    "JetSmart SPA", "K.L.M.", "Lacsa", "Latin American Wings",
    "Oceanair Linhas Aereas", "Plus Ultra Lineas Aereas", "Qantas Airways",
    "Sky Airline", "United Airlines"
}

VALID_TIPOVUELO = {"I", "N"}


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    flights: List[Flight]


app = fastapi.FastAPI()
_model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    for flight in request.flights:
        if flight.OPERA not in VALID_OPERA:
            raise fastapi.HTTPException(status_code=400, detail=f"Unknown OPERA: {flight.OPERA}")
        if flight.TIPOVUELO not in VALID_TIPOVUELO:
            raise fastapi.HTTPException(status_code=400, detail=f"Invalid TIPOVUELO: {flight.TIPOVUELO}")
        if not 1 <= flight.MES <= 12:
            raise fastapi.HTTPException(status_code=400, detail=f"Invalid MES: {flight.MES}")

    data = pd.DataFrame([f.dict() for f in request.flights])
    features = _model.preprocess(data)
    predictions = _model.predict(features)
    return {"predict": predictions}
