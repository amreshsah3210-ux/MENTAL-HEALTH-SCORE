"""
Student Mental Health Score API
--------------------------------
Serves the trained Random Forest pipeline (Mental_health_Model.pkl) and the
frontend in ./static.

Run:
    start_app.bat

Then open http://127.0.0.1:8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("mental-health-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "Mental_health_Model.pkl"))
DATA_PATH = BASE_DIR / "Student Social Media And Mental Health Impact.csv"
FRONTEND_DIR = BASE_DIR / "static"

# Column order used when the pipeline was trained (see notebook).
FEATURE_ORDER = [
    "Study_Hours",
    "Age",
    "Avg_Daily_Usage_Hours",
    "Daily_Unlocks",
    "Physical_Activity_Hours",
    "Sleep_Hours_Per_Night",
    "Stress_Level",
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Purpose_Of_Use",
    "group_countries",
]

# Same list the notebook built with value_counts().index[:10].
# Anything outside it was labelled 'other' (lowercase) during training.
TOP_COUNTRIES = {
    "Other", "India","Nepal", "USA", "Canada", "Australia",
    "UK", "Germany", "Mexico", "Turkey", "France"
}

STRESS_ORDER = ["Low", "Medium", "High", "Very High"]

Platform = Literal[
    "Instagram", "TikTok", "Facebook", "LinkedIn", "YouTube", "Twitter",
    "Snapchat", "WhatsApp", "LINE", "VKontakte", "KakaoTalk", "WeChat",
]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class StudentInput(BaseModel):
    age: int = Field(..., ge=15, le=40)
    gender: Literal["Male", "Female","Other"]
    country: str = Field("Other", max_length=60)
    academic_level: Literal["High School", "Undergraduate", "Graduate","PhD"]
    platform: Platform
    purpose: Literal["Education", "Entertainment", "Networking", "News"]
    daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0, le=1000)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours: float = Field(..., ge=0, le=24)
    stress_level: Literal["Low", "Medium", "High", "Very High"]


class WhatIf(BaseModel):
    label: str
    new_score: float
    change: float


class PredictionOut(BaseModel):
    score: float
    band: str
    message: str
    percentile: Optional[float] = None
    dataset_average: Optional[float] = None
    what_if: list[WhatIf]


# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_error = None
    app.state.scores = None

    try:
        if sklearn.__version__ != "1.6.1":
            raise RuntimeError(
                f"Model requires scikit-learn 1.6.1, but this server is using "
                f"{sklearn.__version__}. Start the app with start_app.bat."
            )
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as exc:  # missing file, version mismatch, etc.
        app.state.model_error = str(exc)
        logger.error("Could not load model at %s: %s", MODEL_PATH, exc)

    if DATA_PATH.exists():
        app.state.scores = (
            pd.read_csv(DATA_PATH, usecols=["Mental_Health_Score"])
            ["Mental_Health_Score"].to_numpy()
        )
    yield


app = FastAPI(title="Student Mental Health Score API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local use; restrict before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_frame(s: StudentInput) -> pd.DataFrame:
    """Turn the request into the exact columns the pipeline was trained on."""
    group = s.country if s.country in TOP_COUNTRIES else "other"
    row = {
        "Study_Hours": s.study_hours,
        "Age": s.age,
        "Avg_Daily_Usage_Hours": s.daily_usage_hours,
        "Daily_Unlocks": s.daily_unlocks,
        "Physical_Activity_Hours": s.physical_activity_hours,
        "Sleep_Hours_Per_Night": s.sleep_hours,
        "Stress_Level": s.stress_level,
        "Gender": s.gender,
        "Academic_Level": s.academic_level,
        "Most_Used_Platform": s.platform,
        "Purpose_Of_Use": s.purpose,
        "group_countries": group,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_many(model, students: list[StudentInput]) -> np.ndarray:
    frame = pd.concat([to_frame(s) for s in students], ignore_index=True)
    return np.clip(model.predict(frame), 0, 10)


def describe(score: float) -> tuple[str, str]:
    if score >= 8:
        return "Strong", "Your habits match the highest-scoring students in the data."
    if score >= 6.5:
        return "Good", "Your habits point to a healthy balance, with some room to improve."
    if score >= 5:
        return "Fair", "The signals are mixed. Small changes to sleep, stress or screen time can help."
    return "Low", (
        "Your habits match students who reported lower scores. If you've felt "
        "low for a while, talking to a counsellor or someone you trust is worth it."
    )


def build_what_if(s: StudentInput, base: float, model) -> list[WhatIf]:
    scenarios: list[tuple[str, StudentInput]] = []

    if s.sleep_hours < 9:
        scenarios.append(("Sleep one more hour", s.model_copy(update={"sleep_hours": min(s.sleep_hours + 1, 10)})))
    if s.daily_usage_hours > 1:
        scenarios.append(("Use social media one hour less", s.model_copy(update={"daily_usage_hours": max(s.daily_usage_hours - 1, 0)})))
    if s.physical_activity_hours < 4:
        scenarios.append(("Add 30 minutes of activity", s.model_copy(update={"physical_activity_hours": s.physical_activity_hours + 0.5})))
    idx = STRESS_ORDER.index(s.stress_level)
    if idx > 0:
        scenarios.append((f"Bring stress down to {STRESS_ORDER[idx - 1].lower()}", s.model_copy(update={"stress_level": STRESS_ORDER[idx - 1]})))

    if not scenarios:
        return []

    preds = predict_many(model, [sc for _, sc in scenarios])
    out = [
        WhatIf(label=label, new_score=round(float(p), 2), change=round(float(p) - base, 2))
        for (label, _), p in zip(scenarios, preds)
    ]
    return sorted(out, key=lambda w: w.change, reverse=True)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/api/health")
def health(request: Request):
    return {"status": "ok" if request.app.state.model is not None else "model_not_loaded"}


@app.post("/api/predict", response_model=PredictionOut)
def predict(student: StudentInput, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model failed to load: {request.app.state.model_error}. "
                "Stop this server and run start_app.bat to use the required "
                "Python 3.12 environment and scikit-learn==1.6.1."
            ),
        )

    try:
        score = round(float(predict_many(model, [student])[0]), 2)
        what_if = build_what_if(student, score, model)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    band, message = describe(score)

    percentile = average = None
    scores = request.app.state.scores
    if scores is not None:
        percentile = round(float((scores < score).mean() * 100), 1)
        average = round(float(scores.mean()), 2)

    return PredictionOut(
        score=score, band=band, message=message,
        percentile=percentile, dataset_average=average, what_if=what_if,
    )


# Frontend (must be mounted last so /api routes win)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")