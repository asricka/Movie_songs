from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .hybrid_recommender import (
        build_user_movie_profile,
        load_movies,
        load_songs,
        recommend_songs,
    )
except ImportError:
    from hybrid_recommender import build_user_movie_profile, load_movies, load_songs, recommend_songs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid movie->music recommendation prototype")
    parser.add_argument(
        "--likes",
        required=True,
        help="Comma-separated movie ids or exact titles (e.g. 'm1,m6' or 'Interstellar,Blade Runner 2049')",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65, help="Weight for lyric/movie similarity")
    parser.add_argument("--beta", type=float, default=0.35, help="Weight for audio mood similarity")
    parser.add_argument("--movies", default="data/movies.json")
    parser.add_argument("--songs", default="data/songs.json")
    return parser


def main() -> None:
    args = _parser().parse_args()
    likes = [x.strip() for x in args.likes.split(",") if x.strip()]

    movies = load_movies(Path(args.movies))
    songs = load_songs(Path(args.songs))

    profile = build_user_movie_profile(movies, likes)
    recs = recommend_songs(
        songs=songs,
        profile=profile,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
    )

    print("Top recommendations")
    for idx, rec in enumerate(recs, start=1):
        print(
            f"{idx}. {rec.song.title} - {rec.song.artist} | "
            f"score={rec.score:.4f} "
            f"(lyric={rec.lyric_similarity:.4f}, audio={rec.audio_similarity:.4f})"
        )


if __name__ == "__main__":
    main()
