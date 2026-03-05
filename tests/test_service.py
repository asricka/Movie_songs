from __future__ import annotations

import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from service import RecommenderService


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RecommenderService.from_files(
            ROOT / "data" / "movies.json", ROOT / "data" / "songs.json"
        )

    def test_movie_catalog_has_expected_items(self) -> None:
        movies = self.service.movie_catalog()
        self.assertGreaterEqual(len(movies), 6)
        self.assertIn("title", movies[0])

    def test_recommendation_payload_shape(self) -> None:
        recs = self.service.recommend_from_movie_likes(
            likes=["Interstellar", "Blade Runner 2049"],
            top_k=3,
            alpha=0.65,
            beta=0.35,
        )
        self.assertEqual(len(recs), 3)
        self.assertIn("song_id", recs[0])
        self.assertIn("score", recs[0])


if __name__ == "__main__":
    unittest.main()
