from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


AUDIO_KEYS = ("valence", "energy", "acousticness", "instrumentalness", "tempo")


@dataclass(frozen=True)
class Movie:
    id: str
    title: str
    genres: List[str]
    plot_embedding: List[float]
    mood_vector: Dict[str, float]


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    artist: str
    lyric_embedding: List[float]
    audio_features: Dict[str, float]


@dataclass(frozen=True)
class Recommendation:
    song: Song
    score: float
    lyric_similarity: float
    audio_similarity: float


def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same dimension")
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _normalize_tempo(tempo: float) -> float:
    # Typical pop tempo range: 60-200 BPM.
    clamped = max(60.0, min(200.0, tempo))
    return (clamped - 60.0) / 140.0


def _audio_vector(features: Dict[str, float]) -> List[float]:
    vector: List[float] = []
    for key in AUDIO_KEYS:
        value = float(features[key])
        if key == "tempo":
            value = _normalize_tempo(value)
        vector.append(value)
    return vector


def _weighted_average(vectors: Iterable[Sequence[float]], weights: Iterable[float]) -> List[float]:
    vectors = list(vectors)
    weights = list(weights)
    if not vectors:
        raise ValueError("Cannot average empty vector list")
    if len(vectors) != len(weights):
        raise ValueError("Vectors and weights length mismatch")
    dims = len(vectors[0])
    for vec in vectors:
        if len(vec) != dims:
            raise ValueError("All vectors must have same dimension")

    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = 1.0
    acc = [0.0] * dims
    for vec, w in zip(vectors, weights):
        for i, value in enumerate(vec):
            acc[i] += value * w
    return [value / total_weight for value in acc]


def load_movies(path: Path) -> List[Movie]:
    raw = json.loads(path.read_text())
    return [
        Movie(
            id=item["id"],
            title=item["title"],
            genres=item["genres"],
            plot_embedding=item["plot_embedding"],
            mood_vector=item["mood_vector"],
        )
        for item in raw
    ]


def load_songs(path: Path) -> List[Song]:
    raw = json.loads(path.read_text())
    return [
        Song(
            id=item["id"],
            title=item["title"],
            artist=item["artist"],
            lyric_embedding=item["lyric_embedding"],
            audio_features=item["audio_features"],
        )
        for item in raw
    ]


def build_user_movie_profile(movies: Sequence[Movie], liked_movie_ids: Sequence[str]) -> Dict[str, List[float]]:
    liked_set = set(liked_movie_ids)
    liked_movies = [m for m in movies if m.id in liked_set or m.title in liked_set]
    if not liked_movies:
        raise ValueError("No liked movies found. Pass movie ids or exact movie titles.")

    # Movie preferences are treated as a stable baseline, so every liked movie is equally weighted.
    weights = [1.0] * len(liked_movies)
    emotion_centroid = _weighted_average((m.plot_embedding for m in liked_movies), weights)
    audio_target = _weighted_average((_audio_vector(m.mood_vector) for m in liked_movies), weights)

    return {
        "emotion_centroid": emotion_centroid,
        "audio_target": audio_target,
    }


def recommend_songs(
    songs: Sequence[Song],
    profile: Dict[str, List[float]],
    top_k: int = 5,
    alpha: float = 0.65,
    beta: float = 0.35,
) -> List[Recommendation]:
    emotion_centroid = profile["emotion_centroid"]
    audio_target = profile["audio_target"]

    results: List[Recommendation] = []
    for song in songs:
        lyric_similarity = _cosine_similarity(song.lyric_embedding, emotion_centroid)
        audio_similarity = _cosine_similarity(_audio_vector(song.audio_features), audio_target)
        score = alpha * lyric_similarity + beta * audio_similarity
        results.append(
            Recommendation(
                song=song,
                score=score,
                lyric_similarity=lyric_similarity,
                audio_similarity=audio_similarity,
            )
        )

    return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
