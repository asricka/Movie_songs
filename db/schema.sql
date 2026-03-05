-- PostgreSQL + pgvector schema for hybrid movie/music recommendation.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movies (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT[] NOT NULL DEFAULT '{}',
    mood_valence REAL NOT NULL,
    mood_energy REAL NOT NULL,
    mood_acousticness REAL NOT NULL,
    mood_instrumentalness REAL NOT NULL,
    mood_tempo REAL NOT NULL,
    plot_embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    valence REAL NOT NULL,
    energy REAL NOT NULL,
    acousticness REAL NOT NULL,
    instrumentalness REAL NOT NULL,
    tempo REAL NOT NULL,
    lyric_embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS movie_swipes (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id TEXT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    direction SMALLINT NOT NULL CHECK (direction IN (-1, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    emotion_centroid vector(1536) NOT NULL,
    audio_target vector(5) NOT NULL,
    alpha REAL NOT NULL DEFAULT 0.65,
    beta REAL NOT NULL DEFAULT 0.35,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS movies_plot_embedding_idx
    ON movies USING ivfflat (plot_embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS songs_lyric_embedding_idx
    ON songs USING ivfflat (lyric_embedding vector_cosine_ops) WITH (lists = 100);

-- Example candidate retrieval: nearest lyrics to a user's emotion centroid.
-- SELECT s.id, s.title
-- FROM songs s
-- JOIN user_profiles u ON u.user_id = $1
-- ORDER BY s.lyric_embedding <=> u.emotion_centroid
-- LIMIT 100;
