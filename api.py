import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Titanic API", description="API predykcji przeżycia na Titanicu")

filename = "model.h5"
model = pickle.load(open(filename, "rb"))

sex_d = {0: "Kobieta", 1: "Mężczyzna"}
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}


class Passenger(BaseModel):
    pclass: int       # 0=Pierwsza, 1=Druga, 2=Trzecia
    sex: int          # 0=Kobieta, 1=Mężczyzna
    age: int
    sibsp: int
    parch: int
    fare: float
    embarked: int     # 0=Cherbourg, 1=Queenstown, 2=Southampton


@app.get("/")
def root():
    return {
        "message": "Titanic API - użyj POST /predict aby uzyskać predykcję",
        "endpoints": ["/predict", "/info", "/docs"],
    }


@app.get("/info")
def info():
    return {"sex": sex_d, "pclass": pclass_d, "embarked": embarked_d}


@app.post("/predict")
def predict(p: Passenger):
    data = [[p.pclass, p.sex, p.age, p.sibsp, p.parch, p.fare, p.embarked]]
    survival = int(model.predict(data)[0])
    confidence = float(model.predict_proba(data)[0][survival])
    return {
        "survived": bool(survival),
        "survived_label": "Tak" if survival == 1 else "Nie",
        "confidence": round(confidence * 100, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
