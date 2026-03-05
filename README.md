# Hybrid Cross-Modal Recommender Prototype

A runnable prototype that bridges movie swipe preferences into music recommendations.

## Recommendation formula

`R(u,s) = alpha * Sim(E_lyrics, E_movie) + beta * Sim(M_audio, M_movieMood)`

- `E_lyrics`: candidate song lyric embedding.
- `E_movie`: centroid of embeddings from movies the user liked.
- `M_audio`: candidate song audio vector (valence, energy, acousticness, instrumentalness, tempo).
- `M_movieMood`: averaged mood vector from liked movies.

## Repo layout

- `src/hybrid_recommender.py`: profile + ranking math.
- `src/service.py`: application service layer.
- `src/app.py`: FastAPI API.
- `src/embedding_provider.py`: pluggable embedding providers (`local`, `openai`).
- `src/ingest.py`: raw text -> embedding helper for ingestion pipelines.
- `src/cli.py`: local CLI for quick scoring.
- `db/schema.sql`: PostgreSQL + pgvector schema.
- `data/movies.json`: sample movie catalog.
- `data/songs.json`: sample song catalog.
- `tests/`: unit tests.

## Quick start

### 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run CLI prototype

```bash
python3 src/cli.py --likes "Interstellar,Blade Runner 2049" --top-k 5
```

### 3) Run API

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

API routes:

- `GET /health`
- `GET /movies`
- `POST /recommend`

Example request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "likes": ["Interstellar", "Blade Runner 2049"],
    "top_k": 5,
    "alpha": 0.65,
    "beta": 0.35
  }'
```

## Embedding providers

Use `.env.example` as a template.

- Local development:
  - `EMBEDDING_PROVIDER=local`
- Real provider:
  - `EMBEDDING_PROVIDER=openai`
  - `OPENAI_API_KEY=...`
  - `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

`src/ingest.py` uses the configured provider for movie plot and lyric text embedding.

## Database schema (pgvector)

Apply `db/schema.sql` to create:

- `movies` with `plot_embedding vector(1536)`
- `songs` with `lyric_embedding vector(1536)`
- `movie_swipes` for Tinder-style likes/dislikes
- `user_profiles` for cached hybrid centroids and weights

## Tests

```bash
python3 -m unittest discover -s tests
```

## GitHub push

After adding your remote:

```bash
git push -u origin main
```

If your default branch is `master`, push that instead:

```bash
git push -u origin master
```
