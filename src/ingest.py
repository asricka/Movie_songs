from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    from .embedding_provider import build_embedding_provider
except ImportError:
    from embedding_provider import build_embedding_provider


@dataclass(frozen=True)
class RawMovie:
    id: str
    title: str
    genres: List[str]
    plot_summary: str
    mood_vector: Dict[str, float]


@dataclass(frozen=True)
class RawSong:
    id: str
    title: str
    artist: str
    lyrics: str
    audio_features: Dict[str, float]


def embed_movie_plot(movie: RawMovie) -> List[float]:
    provider = build_embedding_provider()
    return provider.embed(movie.plot_summary)


def embed_song_lyrics(song: RawSong) -> List[float]:
    provider = build_embedding_provider()
    return provider.embed(song.lyrics)
