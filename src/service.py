from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from .hybrid_recommender import (
        Movie,
        Song,
        build_user_movie_profile,
        load_movies,
        load_songs,
        recommend_songs,
    )
except ImportError:
    from hybrid_recommender import (
        Movie,
        Song,
        build_user_movie_profile,
        load_movies,
        load_songs,
        recommend_songs,
    )


@dataclass
class RecommenderService:
    movies: List[Movie]
    songs: List[Song]

    @classmethod
    def from_files(cls, movies_path: Path, songs_path: Path) -> "RecommenderService":
        return cls(
            movies=load_movies(movies_path),
            songs=load_songs(songs_path),
        )

    def movie_catalog(self) -> List[Dict[str, str]]:
        return [{"id": m.id, "title": m.title} for m in self.movies]

    def recommend_from_movie_likes(
        self,
        likes: Sequence[str],
        top_k: int,
        alpha: float,
        beta: float,
    ) -> List[Dict[str, float | str]]:
        profile = build_user_movie_profile(self.movies, likes)
        recs = recommend_songs(
            songs=self.songs,
            profile=profile,
            top_k=top_k,
            alpha=alpha,
            beta=beta,
        )

        return [
            {
                "song_id": rec.song.id,
                "title": rec.song.title,
                "artist": rec.song.artist,
                "score": round(rec.score, 6),
                "lyric_similarity": round(rec.lyric_similarity, 6),
                "audio_similarity": round(rec.audio_similarity, 6),
            }
            for rec in recs
        ]
