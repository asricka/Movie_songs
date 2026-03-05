from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .service import RecommenderService
except ImportError:
    from service import RecommenderService


class RecommendRequest(BaseModel):
    likes: List[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.65, ge=0.0, le=1.0)
    beta: float = Field(default=0.35, ge=0.0, le=1.0)


class RecommendResponse(BaseModel):
    recommendations: List[dict]


BASE_DIR = Path(__file__).resolve().parents[1]
MOVIES_PATH = Path(os.getenv("MOVIES_PATH", BASE_DIR / "data" / "movies.json"))
SONGS_PATH = Path(os.getenv("SONGS_PATH", BASE_DIR / "data" / "songs.json"))

app = FastAPI(title="Cross-Modal Movie-to-Music Recommender", version="0.1.0")
service = RecommenderService.from_files(MOVIES_PATH, SONGS_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/movies")
def list_movies() -> dict:
    return {"movies": service.movie_catalog()}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    if abs((payload.alpha + payload.beta) - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="alpha + beta must equal 1.0")

    try:
        recommendations = service.recommend_from_movie_likes(
            likes=payload.likes,
            top_k=payload.top_k,
            alpha=payload.alpha,
            beta=payload.beta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RecommendResponse(recommendations=recommendations)
