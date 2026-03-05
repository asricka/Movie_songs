from __future__ import annotations

import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_recommender import build_user_movie_profile, load_movies, load_songs, recommend_songs


ROOT = Path(__file__).resolve().parents[1]


class HybridRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = load_movies(ROOT / "data" / "movies.json")
        self.songs = load_songs(ROOT / "data" / "songs.json")

    def test_build_user_profile_shapes(self) -> None:
        profile = build_user_movie_profile(self.movies, ["Interstellar", "Blade Runner 2049"])
        self.assertEqual(len(profile["emotion_centroid"]), 8)
        self.assertEqual(len(profile["audio_target"]), 5)

    def test_recommendations_return_top_k(self) -> None:
        profile = build_user_movie_profile(self.movies, ["m3", "m6"])
        recs = recommend_songs(self.songs, profile, top_k=3)
        self.assertEqual(len(recs), 3)
        self.assertGreaterEqual(recs[0].score, recs[1].score)
        self.assertGreaterEqual(recs[1].score, recs[2].score)

    def test_romance_movies_favor_soft_tracks(self) -> None:
        profile = build_user_movie_profile(self.movies, ["The Notebook", "La La Land"])
        recs = recommend_songs(self.songs, profile, top_k=1)
        top_title = recs[0].song.title
        self.assertIn(top_title, {"Letters in Rain", "Soft Orbit"})


if __name__ == "__main__":
    unittest.main()
