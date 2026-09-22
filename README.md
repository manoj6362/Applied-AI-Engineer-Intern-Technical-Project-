# 2027 Applied AI Engineer Internship Project

Your work must be your own. You may use AI tools if you list every prompt, in order, in `prompts/prompts.txt`. The first line must name every model you queried (e.g. `Models: GPT 5.6 Sol xhigh`) and each prompt must name the model it went to. Do not include AI outputs.

**The data in this repository is proprietary and confidential. Use it only for this project; do not copy, share, or distribute it.**

### Internship Program Disclosures

* You must be eligible to work in the United States.
* Pay is the greater of your local minimum wage and $13/hour.
* The internship takes place in Spring, Summer, or Fall 2027.

## The Project

You have **72 hours** to submit. Expect roughly **15-20 hours** of focused work. Three parts:

1. **Build the RAG pipeline** (Part 1). This is where your time goes.
2. **Record a short demo** of the provided frontend running on your pipeline (Part 2).
3. **Answer three short written questions** (Part 3).

An incomplete submission that shows your approach beats no submission. Complete Part 3 even if Parts 1 or 2 are unfinished.

## Submission Checklist

Fill in [`SUBMISSION.md`](SUBMISSION.md) (name, application email, final Ollama model digests) and submit:

* [ ] [`SUBMISSION.md`](SUBMISSION.md)
* [ ] [`part1/answers.json`](part1/answers.json), the unedited output of your final `backend/rag.py` run
* [ ] Demo video or link in [`part2`](part2)
* [ ] [`part3/responses.txt`](part3/responses.txt)

---

## Part 1: RAG Backend

The starter ingests six CSVs into PostgreSQL, embeds one document per game with `nomic-embed-text`, retrieves with `pgvector`, and asks `qwen3.5:2b` for a JSON answer shaped by each question's `return` block. Run as shipped, it answers a couple of questions and abstains on the rest.

Data: four structured tables (`game_details`, `player_box_scores`, `players`, `teams`), plus `game_recaps.csv` (a written recap per game) and `injury_notes.csv` (pre-game availability reports). Only games with at least one Western Conference team are included. Extend the pipeline to embed and retrieve the box scores, recaps, and notes (`TODO` blocks in `backend/embed.py`), join and reason over them in `backend/rag.py`, and point `backend/server.py`'s `/api/chat` at your answer path.

`python -m backend.rag` must read `part1/questions.json` at runtime and write `part1/answers.json`. `part1/answers_template.json` shows the shape and `python -m backend.validate` checks it. The schema covers only the public questions; drive your output from each question's `return` block, not from the schema.

### Question tiers

Each of the 11 questions in `questions.json` has a `tier`:

* **`core`** (4): a lookup off the box scores and game details plus one step of reasoning, such as finding a team's first game or its top scorer in a named game. One core question deliberately has no supporting row.
* **`extended`** (4): filters and aggregates over the structured tables, such as a monthly total or a team's largest margin of defeat.
* **`stretch`** (3): facts that exist only in the recap and injury-note text, such as why a player missed a game.

Handle all three tiers through one general retrieve-reason-format flow. Don't special-case questions or tiers.

### Grading

The public questions are examples. Grading swaps in hidden questions of the same kinds with different wording, players, teams, dates, and ids. Don't branch on question ids or hard-code names or values; we read the code and grade mainly on the hidden set.

The core tier is the minimum bar. Similarity search over the starter's game documents alone will not clear it; the core questions need the box scores joined in and a small amount of structured reasoning (an ordinal, a per-game maximum) on top of retrieval.

Grading re-runs your pipeline from scratch on canonical data with the exact model digests below and compares the fresh output to your committed `answers.json`. A mismatch is flagged for review, and the fresh run is what gets scored.

### Answer rules

* The data is the only source of truth. Teams and players are fictional.
* The tables are authoritative. Recaps and injury notes are written from the tables, and a recap can be wrong; when they disagree, the tables win.
* If the data cannot answer a question, return `"answerable": false`, template defaults for the other fields, and an empty `evidence` list. A confident number for a missing row scores zero.
* Cite every source row used, with the keys in the question's `return.evidence` block: `game_details` by `game_id`, `player_box_scores` by `game_id` + `person_id`, `game_recaps` by `recap_id`, `injury_notes` by `note_id`. Cite a recap or note together with the game it describes. For aggregates, cite every contributing row.
* Format scores as `"<winner points>-<loser points>"`, e.g. `"117-102"`.
* Use team mascots (e.g. `"Outlaws"`) and player names exactly as they appear in `teams.csv` and `players.csv`.
* The All-Star break is February 13-18, 2026; "after the break" means on or after February 19.

---

## Part 2: Demo

Point `backend/server.py` at your answer path, run the frontend in `frontend/` against it, and record a screen capture (2 minutes max) of three questions. Include one your pipeline gets wrong or declines if you can; failure cases interest us.

Put the video (`.mp4`, `.mov`, `.webm`) in [`part2`](part2), or an unlisted YouTube/Loom link in `part2/video_link.txt` if it is too large to commit.

---

## Part 3: Written Responses

Answer all three in [`part3/responses.txt`](part3/responses.txt), **200 words or fewer each**. Write these yourself; expect to walk through them in an interview.

1. Pick one public question your pipeline answers correctly and one it gets wrong or abstains on. Quote the value or abstention reason your final `answers.json` holds for the failed one. Explain what retrieval or reasoning decision separates the two, and name the function or query you would change with another day.

2. Pick one data source to add to this pipeline:
   * **Tracking:** [player positional tracking data](https://pr.nba.com/nba-sony-hawk-eye-innovations-partnership/), 29 skeletal points per player (left shoulder, right knee, right ankle, etc.) at 60 frames per second for a full game.
   * **Text:** a large corpus of scouting reports, internal notes, and other documents that shape front-office decisions.

   Name one question a coach or executive could ask that becomes answerable only with that source. Describe how you would represent, retrieve, and validate it, and where the current `embed.py` / `rag.py` design would break.

3. Given the opportunity to work with any group in Basketball Operations (coaching, performance, scouting, strategy, etc.), which would you be most excited to work with and why?

---

## Quick Start

1. Install and open [Docker Desktop](https://www.docker.com/get-started/) and [Ollama](https://ollama.com/download).
2. Pull the models (about 3 GB) while the app builds:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:2b

docker compose up -d db
docker compose build app
```

Ollama runs natively so it can use your GPU; the app reaches it at `host.docker.internal:11434`.

### Run Part 1

```bash
docker compose run --rm app python -m backend.ingest
docker compose run --rm app python -m backend.embed
docker compose run --rm app python -m backend.rag
```

Embedding takes a while; smoke-test with `python -m backend.embed --limit 50` and batch requests with `ollama_embed_batch` in `backend/utils.py`. Ingest replaces the tables on each run. `docker compose down -v` wipes the database, embeddings included.

### Run Part 2

```bash
docker compose run --rm --service-ports app uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
```

Then in `frontend/` (Node 18, 20, or 22): `npm install && npm start`, and open `http://localhost:4200/`.

### Models

Grading uses exactly these digests; check yours with `ollama list` and record them in `SUBMISSION.md`:

* `nomic-embed-text`: `0a109f422b47`
* `qwen3.5:2b`: `324d162be6ca`

`backend/utils.py` sends `"think": false`, `temperature` 0, and a fixed `seed` so reruns reproduce your answers; keep that. Retrieval is an exact scan with a stable `ORDER BY` for the same reason; an approximate index can make the graded rerun differ from your committed answers. Keep model names in `EMBED_MODEL` and `LLM_MODEL`.

### Docker Ollama fallback

If you cannot run Ollama natively:

```bash
docker compose --profile docker-ollama up -d db ollama
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull qwen3.5:2b
OLLAMA_HOST=http://ollama:11434 docker compose --profile docker-ollama build app
```

Use `OLLAMA_HOST=http://ollama:11434` and the `docker-ollama` profile for every later command.

---

Questions?
Email datasolutions@okcthunder.com
