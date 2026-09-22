"""Answer part1/questions.json with retrieval-augmented generation.

For each question: extract entities from the live teams/players tables, retrieve
a working set of game_details / box scores / recaps / notes (structured filters
plus pgvector), reason over that set, and format a JSON object matching the
question's `return` block. Writes part1/answers.json as a list of {id, result}.

usage:
    python -m backend.rag
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))

TOP_K = 8
QUERY_PREFIX = "search_query: "

# README: All-Star break is February 13-18, 2026; "after the break" means on or after February 19.
AFTER_BREAK = date(2026, 2, 19)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

GO_AHEAD_PATTERNS = [
    re.compile(
        r"go-ahead (?:bucket|basket|shot|jumper|score|layup|points).{0,40}from "
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)",
        re.I,
    ),
    re.compile(
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)\s+"
        r"(?:hit|provided|delivered|supplied|put).*?go-ahead",
        re.I,
    ),
    re.compile(
        r"it was ([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?) who put .{0,40}ahead",
        re.I,
    ),
    re.compile(
        r"([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+)?)(?:'s) go-ahead",
        re.I,
    ),
]

DEFICIT_PATTERNS = [
    re.compile(
        r"(?:rallied back from|overcame|erased|facing|faced|down by)\s+"
        r"(?:a |an )?(\d+)[- ]point (?:deficit|hole)",
        re.I,
    ),
    re.compile(r"(?:down|trailing|trailed) by (?:as many as )?(\d+) points", re.I),
    re.compile(r"trailing by (\d+) points earlier", re.I),
    re.compile(r"(\d+)[- ]point (?:deficit|hole)", re.I),
]


@dataclass
class Catalog:
    games: pd.DataFrame
    boxes: pd.DataFrame
    recaps: pd.DataFrame
    notes: pd.DataFrame
    teams: list[dict]
    players: list[dict]


@dataclass
class Entities:
    team_ids: list[int] = field(default_factory=list)
    player_ids: list[int] = field(default_factory=list)
    on_date: date | None = None
    month: int | None = None
    year: int | None = None
    after_break: bool = False
    opener: bool = False
    lost: bool = False
    won: bool = False
    starter: bool = False
    largest: bool = False
    most: bool = False
    against: bool = False
    margin_at_least: int | None = None
    points_at_least: int | None = None


@dataclass
class WorkingSet:
    games: pd.DataFrame
    boxes: pd.DataFrame
    recaps: pd.DataFrame
    notes: pd.DataFrame
    vector_rows: list[dict] = field(default_factory=list)


def retrieve(cx: Connection, qvec: list[float], k: int = TOP_K) -> list[dict]:
    """Retrieve the k nearest game_details documents to a query embedding.

    Shared by rag.py and server.py so both use the exact same retrieval query
    and column set (see server.py: import from here instead of duplicating).

    :param cx: open SQLAlchemy connection
    :param qvec: query embedding, same dimensionality as game_details.embedding
    :param k: number of rows to retrieve
    :return: rows with game_id, doc, and cosine similarity score, best first
    """
    sql = (
        "SELECT game_id, doc, 1 - (embedding <=> (:q)::vector) AS score "
        "FROM game_details WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> (:q)::vector, game_id LIMIT :k"
    )
    return [dict(r) for r in cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()]


def retrieve_table(cx: Connection, table: str, id_cols: list[str], qvec: list[float], k: int) -> list[dict]:
    """Exact cosine scan of one embedded table, with a stable ORDER BY tie-break."""
    id_select = ", ".join(id_cols)
    order = ", ".join(id_cols)
    sql = (
        f"SELECT {id_select}, doc, 1 - (embedding <=> (:q)::vector) AS score "
        f"FROM {table} WHERE embedding IS NOT NULL "
        f"ORDER BY embedding <=> (:q)::vector, {order} LIMIT :k"
    )
    try:
        return [dict(r) for r in cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()]
    except Exception:
        return []


def build_context(rows: list[dict]) -> str:
    """Render retrieved rows as the context block shown to the LLM."""
    lines = []
    for r in rows:
        table = r.get("table", "game_details")
        if table == "player_box_scores":
            tag = f"[player_box_scores game_id={r.get('game_id')} person_id={r.get('person_id')}]"
        elif table == "game_recaps":
            tag = f"[game_recaps id={r.get('recap_id', r.get('id'))}]"
        elif table == "injury_notes":
            tag = f"[injury_notes id={r.get('note_id', r.get('id'))}]"
        else:
            tag = f"[game_details id={r.get('game_id', r.get('id'))}]"
        lines.append(f"{tag} {r.get('doc', '')}")
    return "\n".join(lines)


def coerce_type(value, type_name: str):
    """Coerce a raw JSON value to the type named in a question's `return` block."""
    if type_name == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if type_name == "int":
        return int(round(float(value)))
    if type_name == "str":
        return str(value)
    raise ValueError(f"Unknown type in return schema: {type_name}")


def extract_json(raw: str) -> dict | None:
    """Pull the first top-level JSON object out of an LLM's free-text response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def ground_evidence(parsed_evidence, declared_shapes: list[dict], retrieved: dict) -> list[dict]:
    """Keep only evidence rows the model returned that we can actually verify."""
    if not isinstance(parsed_evidence, list):
        return []
    grounded = []
    seen = set()
    for entry in parsed_evidence:
        if not isinstance(entry, dict):
            continue
        shape = next((s for s in declared_shapes if s.get("table") == entry.get("table")), None)
        if shape is None:
            continue
        try:
            item = {"table": entry["table"]}
            for key, type_name in shape.items():
                if key != "table":
                    item[key] = coerce_type(entry[key], type_name)
        except (KeyError, TypeError, ValueError):
            continue
        table = item["table"]
        if table == "game_details" and item["id"] in retrieved.get("game_details", set()):
            key = (table, item["id"])
        elif table == "player_box_scores":
            pair = (item["game_id"], item["person_id"])
            if pair not in retrieved.get("player_box_scores", set()):
                continue
            key = (table, pair)
        elif table == "game_recaps" and item["id"] in retrieved.get("game_recaps", set()):
            key = (table, item["id"])
        elif table == "injury_notes" and item["id"] in retrieved.get("injury_notes", set()):
            key = (table, item["id"])
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        grounded.append(item)
    return grounded


def load_teams(cx: Connection) -> list[dict]:
    """Load team_id, city, name, and abbreviation for every team, once per run."""
    sql = "SELECT team_id, city, name, abbreviation FROM teams"
    return [dict(r) for r in cx.execute(text(sql)).mappings().all()]


def load_catalog(cx: Connection) -> Catalog:
    """Load the structured tables once so every question shares the same snapshot."""
    teams = load_teams(cx)
    players = [dict(r) for r in cx.execute(text(
        "SELECT player_id, team_id, first_name, last_name FROM players"
    )).mappings().all()]
    games = pd.read_sql(text(
        "SELECT g.game_id, g.season, g.game_timestamp, g.home_team_id, g.away_team_id, "
        "g.home_points, g.away_points, g.winning_team_id, "
        "home.name AS home_name, away.name AS away_name, "
        "win.name AS winner_name, "
        "CASE WHEN g.winning_team_id = g.home_team_id THEN away.name ELSE home.name END AS loser_name "
        "FROM game_details g "
        "JOIN teams home ON home.team_id = g.home_team_id "
        "JOIN teams away ON away.team_id = g.away_team_id "
        "JOIN teams win ON win.team_id = g.winning_team_id"
    ), cx)
    games["game_timestamp"] = pd.to_datetime(games["game_timestamp"], utc=True)
    games["game_date"] = games["game_timestamp"].dt.date
    boxes = pd.read_sql(text(
        "SELECT b.game_id, b.person_id, b.team_id, b.starter, b.points, "
        "p.first_name || ' ' || p.last_name AS player_name, p.last_name, p.first_name "
        "FROM player_box_scores b JOIN players p ON p.player_id = b.person_id"
    ), cx)
    boxes["starter_flag"] = boxes["starter"].map(
        lambda v: str(v).strip().lower() in {"true", "t", "1"}
    )
    recaps = pd.read_sql(text("SELECT recap_id, game_id, text FROM game_recaps"), cx)
    notes = pd.read_sql(text(
        "SELECT n.note_id, n.game_id, n.player_id, n.team_id, n.note_date, n.status, n.text, "
        "p.first_name || ' ' || p.last_name AS player_name "
        "FROM injury_notes n JOIN players p ON p.player_id = n.player_id"
    ), cx)
    notes["note_date"] = pd.to_datetime(notes["note_date"]).dt.date
    return Catalog(games=games, boxes=boxes, recaps=recaps, notes=notes, teams=teams, players=players)


def normalize_team_names(result: dict, teams: list[dict]) -> dict:
    """Force team references in string fields down to the mascot name."""
    skip_fields = {"evidence", "player_name", "reason", "status"}
    normalized = dict(result)
    for key, value in result.items():
        if key in skip_fields or not isinstance(value, str):
            continue
        matched_mascots = {
            team["name"]
            for team in teams
            if re.search(rf"\b{re.escape(team['city'] + ' ' + team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['name'])}\b", value)
            or re.search(rf"\b{re.escape(team['abbreviation'])}\b", value)
        }
        if len(matched_mascots) == 1:
            normalized[key] = next(iter(matched_mascots))
    return normalized


def normalize_player_names(result: dict, catalog: Catalog, person_ids: set[int] | None = None) -> dict:
    """Rewrite player_name to the canonical 'First Last' form from players.csv."""
    if "player_name" not in result or not isinstance(result.get("player_name"), str):
        return result
    value = result["player_name"].strip()
    if not value:
        return result
    players = catalog.players
    if person_ids:
        players = [p for p in players if int(p["player_id"]) in person_ids]
    full = {f"{p['first_name']} {p['last_name']}": p for p in players}
    if value in full:
        result["player_name"] = value
        return result
    last_hits = [f"{p['first_name']} {p['last_name']}" for p in players if p["last_name"].lower() == value.lower()]
    if len(last_hits) == 1:
        result["player_name"] = last_hits[0]
        return result
    for name in sorted(full, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", value, re.I):
            result["player_name"] = name
            return result
    return result


def abstain_result(return_spec: dict) -> dict:
    """Build a schema-valid abstention result straight from a question's `return` block."""
    zeros = {"bool": False, "int": 0, "str": ""}
    result = {key: zeros[type_name] for key, type_name in return_spec.items() if key != "evidence"}
    result["answerable"] = False
    result["evidence"] = []
    return result


def evidence_fully_grounded(declared_shapes: list[dict], grounded: list[dict]) -> bool:
    """Check that every evidence table the question requires has a grounded row."""
    declared_tables = {s["table"] for s in declared_shapes}
    grounded_tables = {g["table"] for g in grounded}
    return bool(declared_tables) and declared_tables.issubset(grounded_tables)


def parse_entities(question: str, catalog: Catalog) -> Entities:
    """Match teams, players, dates, and filter cues from the question text only."""
    ent = Entities()
    q = question
    ql = question.lower()

    team_hits: list[tuple[int, int, int]] = []  # start, end, team_id
    for team in catalog.teams:
        tid = int(team["team_id"])
        needles = (team["city"] + " " + team["name"], team["name"], team["abbreviation"])
        for needle in sorted(needles, key=len, reverse=True):
            for m in re.finditer(rf"\b{re.escape(needle)}\b", q, re.I):
                team_hits.append((m.start(), m.end(), tid))
    team_hits.sort()
    used_spans = []
    for start, end, tid in team_hits:
        if any(start < e and end > s for s, e in used_spans):
            continue
        used_spans.append((start, end))
        if tid not in ent.team_ids:
            ent.team_ids.append(tid)

    players_sorted = sorted(
        catalog.players, key=lambda p: len(f"{p['first_name']} {p['last_name']}"), reverse=True
    )
    for player in players_sorted:
        full = f"{player['first_name']} {player['last_name']}"
        if re.search(rf"\b{re.escape(full)}\b", q, re.I):
            pid = int(player["player_id"])
            if pid not in ent.player_ids:
                ent.player_ids.append(pid)

    dm = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),\s*(\d{4})\b",
        q,
        re.I,
    )
    if dm:
        ent.on_date = date(int(dm.group(3)), MONTHS[dm.group(1).lower()], int(dm.group(2)))
    else:
        mm = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+(\d{4})\b",
            q,
            re.I,
        )
        if mm:
            ent.month = MONTHS[mm.group(1).lower()]
            ent.year = int(mm.group(2))

    ent.after_break = bool(re.search(r"after the (all[ -]?star )?break", ql))
    ent.opener = bool(re.search(r"\b(season opener|first game|opener)\b", ql))
    ent.lost = bool(re.search(r"\b(lost|loss|defeat|defeated them)\b", ql))
    ent.won = bool(re.search(r"\b(win|won|victory)\b", ql))
    ent.starter = "starter" in ql
    ent.largest = "largest" in ql
    ent.most = bool(re.search(r"\bmost\b", ql))
    ent.against = bool(re.search(r"\bagainst\b|\bvs\.?\b", ql))
    m_margin = re.search(r"\bby (\d+)\s+or more\b", ql)
    if m_margin:
        ent.margin_at_least = int(m_margin.group(1))
    m_pts = re.search(r"\b(\d+)\s+or more points\b", ql)
    if m_pts:
        ent.points_at_least = int(m_pts.group(1))
    return ent


def _game_involves(games: pd.DataFrame, team_id: int) -> pd.Series:
    return (games["home_team_id"] == team_id) | (games["away_team_id"] == team_id)


def _margin_for_team(row, team_id: int) -> int:
    if int(row.home_team_id) == team_id:
        return int(row.home_points) - int(row.away_points)
    return int(row.away_points) - int(row.home_points)


def _score_str(row) -> str:
    if int(row.winning_team_id) == int(row.home_team_id):
        return f"{int(row.home_points)}-{int(row.away_points)}"
    return f"{int(row.away_points)}-{int(row.home_points)}"


def _opponent_name(row, team_id: int) -> str:
    if int(row.home_team_id) == team_id:
        return row.away_name
    return row.home_name


def _is_win_for(row, team_id: int) -> bool:
    return int(row.winning_team_id) == int(team_id)


def filter_games(catalog: Catalog, ent: Entities) -> pd.DataFrame:
    games = catalog.games.copy()
    if ent.on_date is not None:
        games = games[games["game_date"] == ent.on_date]
    if ent.month is not None and ent.year is not None:
        ts = games["game_timestamp"]
        games = games[(ts.dt.month == ent.month) & (ts.dt.year == ent.year)]
    if ent.after_break:
        games = games[games["game_date"] >= AFTER_BREAK]
    if len(ent.team_ids) >= 2:
        a, b = ent.team_ids[0], ent.team_ids[1]
        games = games[_game_involves(games, a) & _game_involves(games, b)]
    elif len(ent.team_ids) == 1:
        games = games[_game_involves(games, ent.team_ids[0])]
    if ent.opener and len(ent.team_ids) == 1:
        subset = catalog.games[_game_involves(catalog.games, ent.team_ids[0])]
        if not subset.empty:
            first_id = subset.sort_values(["game_timestamp", "game_id"]).iloc[0]["game_id"]
            games = catalog.games[catalog.games["game_id"] == first_id]
    if ent.lost and len(ent.team_ids) == 1:
        tid = ent.team_ids[0]
        games = games[games["winning_team_id"] != tid]
        if ent.margin_at_least is not None:
            games = games[games.apply(lambda r: abs(_margin_for_team(r, tid)) >= ent.margin_at_least, axis=1)]
    elif ent.won and len(ent.team_ids) == 1 and not ent.lost:
        games = games[games["winning_team_id"] == ent.team_ids[0]]
    return games.reset_index(drop=True)


def filter_boxes(catalog: Catalog, games: pd.DataFrame, ent: Entities) -> pd.DataFrame:
    boxes = catalog.boxes
    if not games.empty:
        boxes = boxes[boxes["game_id"].isin(games["game_id"])]
    if ent.player_ids:
        player_boxes = catalog.boxes[catalog.boxes["person_id"].isin(ent.player_ids)]
        if ent.on_date is not None or (ent.month is not None and ent.year is not None) or ent.after_break or ent.opener:
            if not games.empty:
                player_boxes = player_boxes[player_boxes["game_id"].isin(games["game_id"])]
            else:
                g = catalog.games
                if ent.on_date is not None:
                    g = g[g["game_date"] == ent.on_date]
                if ent.month is not None and ent.year is not None:
                    player_boxes = player_boxes.merge(
                        catalog.games[["game_id", "game_timestamp"]], on="game_id", how="left"
                    )
                    ts = pd.to_datetime(player_boxes["game_timestamp"], utc=True)
                    player_boxes = player_boxes[(ts.dt.month == ent.month) & (ts.dt.year == ent.year)]
        # Player + opponent team: keep the player's rows in games vs that opponent.
        if ent.against and len(ent.team_ids) == 1 and not games.empty:
            player_boxes = player_boxes[player_boxes["game_id"].isin(games["game_id"])]
        boxes = player_boxes
    if ent.starter:
        if len(ent.team_ids) == 1:
            boxes = boxes[(boxes["team_id"] == ent.team_ids[0]) & (boxes["starter_flag"])]
        else:
            boxes = boxes[boxes["starter_flag"]]
    if ent.points_at_least is not None:
        # "against TEAM" means the shooter's opponent is TEAM, not the shooter's own team.
        if ent.against and len(ent.team_ids) == 1:
            boxes = boxes[boxes["team_id"] != ent.team_ids[0]]
        boxes = boxes[boxes["points"] >= ent.points_at_least]
    return boxes.reset_index(drop=True)


def parsed_deficits(recap_text: str) -> list[int]:
    values = []
    for pat in DEFICIT_PATTERNS:
        for m in pat.finditer(recap_text):
            values.append(int(m.group(1)))
    return values


def parsed_go_ahead_names(recap_text: str) -> list[str]:
    names = []
    for pat in GO_AHEAD_PATTERNS:
        for m in pat.finditer(recap_text):
            names.append(m.group(1).strip())
    return names


def resolve_player_token(token: str, candidates: pd.DataFrame) -> str | None:
    if candidates.empty:
        return None
    token_l = token.lower()
    full_hits = candidates[candidates["player_name"].str.lower() == token_l]
    if len(full_hits) == 1:
        return full_hits.iloc[0]["player_name"]
    last_hits = candidates[candidates["last_name"].str.lower() == token_l]
    if len(last_hits) == 1:
        return last_hits.iloc[0]["player_name"]
    contains = candidates[candidates["player_name"].str.lower().str.contains(re.escape(token_l), regex=True)]
    if len(contains) == 1:
        return contains.iloc[0]["player_name"]
    return None


def evidence_from_frames(games: pd.DataFrame, boxes: pd.DataFrame, recaps: pd.DataFrame, notes: pd.DataFrame,
                         declared_shapes: list[dict]) -> list[dict]:
    wanted = {s["table"] for s in declared_shapes}
    out: list[dict] = []
    if "game_details" in wanted:
        for gid in sorted(int(x) for x in games["game_id"].unique()):
            out.append({"table": "game_details", "id": gid})
    if "player_box_scores" in wanted:
        pairs = boxes[["game_id", "person_id"]].drop_duplicates().sort_values(["game_id", "person_id"])
        for row in pairs.itertuples(index=False):
            out.append({"table": "player_box_scores", "game_id": int(row.game_id), "person_id": int(row.person_id)})
    if "game_recaps" in wanted:
        for rid in sorted(int(x) for x in recaps["recap_id"].unique()):
            out.append({"table": "game_recaps", "id": rid})
    if "injury_notes" in wanted:
        for nid in sorted(int(x) for x in notes["note_id"].unique()):
            out.append({"table": "injury_notes", "id": nid})
    return out


def retrieved_index(games, boxes, recaps, notes) -> dict:
    return {
        "game_details": set(int(x) for x in games["game_id"].tolist()) if not games.empty else set(),
        "player_box_scores": set(
            (int(r.game_id), int(r.person_id)) for r in boxes.itertuples(index=False)
        ) if not boxes.empty else set(),
        "game_recaps": set(int(x) for x in recaps["recap_id"].tolist()) if not recaps.empty else set(),
        "injury_notes": set(int(x) for x in notes["note_id"].tolist()) if not notes.empty else set(),
    }


def derive_from_tables(question: str, return_spec: dict, catalog: Catalog, ws: WorkingSet, ent: Entities) -> dict | None:
    """Reason over the working set. Tables win over recaps for scores and names."""
    keys = set(return_spec.keys())
    games, boxes, recaps, notes = ws.games, ws.boxes, ws.recaps, ws.notes
    declared = return_spec.get("evidence", [])

    # Injury / availability: status + reason live only in notes.
    if keys >= {"status", "reason"}:
        n = notes
        if ent.player_ids:
            n = n[n["player_id"].isin(ent.player_ids)]
        if not games.empty:
            n = n[n["game_id"].isin(games["game_id"])]
        if n.empty:
            return None
        n = n.sort_values(["note_id"])
        row = n.iloc[0]
        g = catalog.games[catalog.games["game_id"] == row["game_id"]]
        ev_games = g if not g.empty else games
        result = {
            "answerable": True,
            "status": str(row["status"]),
            "reason": str(row["text"]),
            "evidence": evidence_from_frames(ev_games, boxes.iloc[0:0], recaps.iloc[0:0], n.iloc[[0]], declared),
        }
        return result

    # Narrative go-ahead (player_name only, recap required).
    if keys >= {"player_name"} and "points" not in keys and "games" not in keys and any(
        s.get("table") == "game_recaps" for s in declared
    ):
        if games.empty or recaps.empty:
            return None
        recaps = recaps.merge(games[["game_id"]], on="game_id", how="inner")
        if recaps.empty:
            return None
        recaps = recaps.sort_values("recap_id")
        game_row = games.sort_values(["game_timestamp", "game_id"]).iloc[0]
        recap_row = recaps[recaps["game_id"] == game_row["game_id"]]
        if recap_row.empty:
            recap_row = recaps.iloc[[0]]
            game_row = games[games["game_id"] == recap_row.iloc[0]["game_id"]].iloc[0]
        tokens = parsed_go_ahead_names(str(recap_row.iloc[0]["text"]))
        gboxes = catalog.boxes[catalog.boxes["game_id"] == game_row["game_id"]]
        if ent.team_ids:
            # Prefer the mentioned team's players when the question names a winner/side.
            side = gboxes[gboxes["team_id"] == ent.team_ids[0]]
            if not side.empty:
                gboxes = side
        resolved = None
        for tok in tokens:
            resolved = resolve_player_token(tok, gboxes)
            if resolved:
                break
        if not resolved:
            return None
        hit = gboxes[gboxes["player_name"] == resolved]
        if hit.empty:
            return None
        ev_boxes = hit
        ev_games = catalog.games[catalog.games["game_id"] == game_row["game_id"]]
        ev_recaps = recap_row
        return {
            "answerable": True,
            "player_name": resolved,
            "evidence": evidence_from_frames(ev_games, ev_boxes, ev_recaps, notes.iloc[0:0], declared),
        }

    # Largest deficit overcome (recap + game score from the table).
    if "deficit" in keys:
        if games.empty or recaps.empty:
            return None
        recaps = recaps.merge(games[["game_id", "home_team_id", "away_team_id", "winning_team_id",
                                     "home_points", "away_points", "home_name", "away_name", "winner_name"]],
                              on="game_id", how="inner")
        best = None
        for row in recaps.itertuples(index=False):
            if ent.team_ids:
                tid = ent.team_ids[0]
                if int(row.winning_team_id) != tid:
                    continue
            deficits = parsed_deficits(str(row.text))
            if not deficits:
                continue
            d = max(deficits)
            if best is None or d > best[0]:
                best = (d, row)
        if best is None:
            return None
        d, row = best
        g = catalog.games[catalog.games["game_id"] == row.game_id]
        r = catalog.recaps[catalog.recaps["recap_id"] == row.recap_id]
        opponent = _opponent_name(g.iloc[0], ent.team_ids[0]) if ent.team_ids else g.iloc[0]["loser_name"]
        return {
            "answerable": True,
            "opponent": opponent,
            "deficit": d,
            "score": _score_str(g.iloc[0]),
            "evidence": evidence_from_frames(g, boxes.iloc[0:0], r, notes.iloc[0:0], declared),
        }

    # Largest margin of defeat / similar game-level extremum.
    if "margin" in keys:
        if games.empty:
            return None
        tid = ent.team_ids[0] if ent.team_ids else None
        if tid is None:
            return None
        g = games.copy()
        g["_margin"] = g.apply(lambda r: abs(_margin_for_team(r, tid)), axis=1)
        if ent.lost or "defeat" in question.lower():
            g = g[g["winning_team_id"] != tid]
        if g.empty:
            return None
        g = g.sort_values(["_margin", "game_id"], ascending=[False, True])
        row = g.iloc[0]
        pick = catalog.games[catalog.games["game_id"] == row["game_id"]]
        return {
            "answerable": True,
            "opponent": _opponent_name(row, tid),
            "margin": int(row["_margin"]),
            "score": _score_str(row),
            "evidence": evidence_from_frames(pick, boxes.iloc[0:0], recaps.iloc[0:0], notes.iloc[0:0], declared),
        }

    # Final score / winner for a specific game.
    if keys >= {"winner", "score"}:
        if games.empty:
            return None
        if len(games) != 1:
            games = games.sort_values(["game_timestamp", "game_id"])
            if ent.on_date is None and len(games) != 1:
                return None
            if len(games) != 1:
                return None
        row = games.iloc[0]
        return {
            "answerable": True,
            "winner": row["winner_name"],
            "score": _score_str(row),
            "evidence": evidence_from_frames(games.iloc[[0]], boxes.iloc[0:0], recaps.iloc[0:0], notes.iloc[0:0], declared),
        }

    # Count of qualifying games for a player (e.g. 30-point games).
    if keys >= {"player_name", "games"}:
        if boxes.empty:
            return None
        grouped = boxes.groupby(["person_id", "player_name"], as_index=False).agg(
            games=("game_id", "nunique"), points=("points", "sum")
        )
        grouped = grouped.sort_values(["games", "person_id"], ascending=[False, True])
        row = grouped.iloc[0]
        used = boxes[boxes["person_id"] == row["person_id"]]
        used_games = catalog.games[catalog.games["game_id"].isin(used["game_id"])]
        return {
            "answerable": True,
            "player_name": row["player_name"],
            "games": int(row["games"]),
            "evidence": evidence_from_frames(used_games, used, recaps.iloc[0:0], notes.iloc[0:0], declared),
        }

    # Named player + points, or "who scored the most" + points.
    if "points" in keys:
        if boxes.empty:
            return None
        if ent.player_ids:
            used = boxes[boxes["person_id"].isin(ent.player_ids)]
            if used.empty:
                return None
            points = int(used["points"].sum())
            result = {"answerable": True, "points": points}
            if "games" in keys:
                result["games"] = int(used["game_id"].nunique())
            if "player_name" in keys:
                result["player_name"] = used.iloc[0]["player_name"]
            used_games = catalog.games[catalog.games["game_id"].isin(used["game_id"])]
            result["evidence"] = evidence_from_frames(used_games, used, recaps.iloc[0:0], notes.iloc[0:0], declared)
            return result

        if ent.team_ids and (boxes["team_id"] == ent.team_ids[0]).any():
            boxes = boxes[boxes["team_id"] == ent.team_ids[0]]
        grouped = boxes.groupby(["person_id", "player_name"], as_index=False).agg(
            points=("points", "sum"), games=("game_id", "nunique")
        )
        grouped = grouped.sort_values(["points", "person_id"], ascending=[False, True])
        row = grouped.iloc[0]
        used = boxes[boxes["person_id"] == row["person_id"]]
        used_games = catalog.games[catalog.games["game_id"].isin(used["game_id"])]
        result = {"answerable": True, "points": int(row["points"])}
        if "player_name" in keys:
            result["player_name"] = row["player_name"]
        if "games" in keys:
            result["games"] = int(row["games"])
        result["evidence"] = evidence_from_frames(used_games, used, recaps.iloc[0:0], notes.iloc[0:0], declared)
        return result

    return None


def verify_numeric(result: dict, catalog: Catalog) -> dict | None:
    """Re-derive claimed box-score numbers from cited rows; tables win."""
    if not result.get("answerable"):
        return result
    box_ev = [e for e in result.get("evidence", []) if e.get("table") == "player_box_scores"]
    if box_ev and "points" in result:
        pairs = {(e["game_id"], e["person_id"]) for e in box_ev}
        subset = catalog.boxes[catalog.boxes.apply(lambda r: (int(r.game_id), int(r.person_id)) in pairs, axis=1)]
        if subset.empty:
            return None
        total = int(subset["points"].sum())
        if int(result["points"]) != total:
            result["points"] = total
        if "games" in result:
            result["games"] = int(subset["game_id"].nunique())
        names = subset["player_name"].unique().tolist()
        if "player_name" in result and len(names) == 1:
            result["player_name"] = names[0]
    game_ev = [e for e in result.get("evidence", []) if e.get("table") == "game_details"]
    if game_ev and "score" in result and len(game_ev) == 1:
        g = catalog.games[catalog.games["game_id"] == game_ev[0]["id"]]
        if not g.empty:
            result["score"] = _score_str(g.iloc[0])
            if "winner" in result:
                result["winner"] = g.iloc[0]["winner_name"]
    return result


def llm_fill(question: dict, ctx: str, return_spec: dict, teams: list[dict]) -> dict | None:
    prompt = (
        "You answer questions about NBA-style games using only the context below. "
        "The structured tables are authoritative; recaps can be wrong — if they disagree, trust the tables. "
        "Respond with a single JSON object matching this shape (types shown, not values): "
        f"{json.dumps(return_spec)}\n"
        "Every evidence row you cite must be one you can see in the context below. "
        "If the context doesn't contain enough information to answer, set \"answerable\": false "
        "and return your best-effort defaults for the other fields with an empty evidence list. "
        "Use team mascots only, e.g. \"Outlaws\" instead of \"Spokane Outlaws (SPO)\". "
        "Give player names exactly as \"First Last\". "
        "Format any score as \"<winner points>-<loser points>\", e.g. \"117-102\". "
        "The All-Star break is February 13-18, 2026; after the break means on or after February 19.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question['question']}\nJSON:"
    )
    raw = ollama_generate(LLM_MODEL, prompt)
    parsed = extract_json(raw)
    if parsed is None:
        return None
    result = {}
    for key, type_name in return_spec.items():
        if key == "evidence":
            continue
        if key not in parsed:
            return None
        try:
            result[key] = coerce_type(parsed[key], type_name)
        except (TypeError, ValueError):
            return None
    result["_raw_evidence"] = parsed.get("evidence")
    result = normalize_team_names(result, teams)
    return result


def compact_context(ws: WorkingSet, catalog: Catalog, ent: Entities, limit_games: int = 40) -> str:
    lines = []
    games = ws.games.sort_values(["game_timestamp", "game_id"])
    if len(games) > limit_games:
        games = games.tail(limit_games)
    for row in games.itertuples(index=False):
        lines.append(
            f"[game_details id={int(row.game_id)}] {row.game_date}: {row.home_name} {int(row.home_points)}-"
            f"{int(row.away_points)} {row.away_name}, winner {row.winner_name}"
        )
    boxes = ws.boxes
    if len(boxes) > 80:
        boxes = boxes.sort_values("points", ascending=False).head(80)
    for row in boxes.itertuples(index=False):
        starter = "starter" if row.starter_flag else "bench"
        lines.append(
            f"[player_box_scores game_id={int(row.game_id)} person_id={int(row.person_id)}] "
            f"{row.player_name} ({starter}) {int(row.points)} points"
        )
    for row in ws.recaps.head(12).itertuples(index=False):
        lines.append(f"[game_recaps id={int(row.recap_id)}] {str(row.text)[:1200]}")
    for row in ws.notes.head(12).itertuples(index=False):
        lines.append(
            f"[injury_notes id={int(row.note_id)}] {row.player_name} status={row.status}: {row.text}"
        )
    return "\n".join(lines)


def build_working_set(cx: Connection, question_text: str, catalog: Catalog, ent: Entities,
                      return_spec: dict) -> WorkingSet:
    games = filter_games(catalog, ent)
    # Player+date/month with no team: restrict games to those the player actually has boxes in,
    # unless this is the planted missing-row case (keep the dated/matchup games for abstention).
    if ent.player_ids and (ent.month is not None or ent.on_date is not None) and not ent.against:
        pb = catalog.boxes[catalog.boxes["person_id"].isin(ent.player_ids)]
        if not games.empty:
            pb = pb[pb["game_id"].isin(games["game_id"])]
        else:
            pb = pb.merge(catalog.games[["game_id", "game_timestamp", "game_date"]], on="game_id", how="left")
            if ent.month is not None and ent.year is not None:
                ts = pd.to_datetime(pb["game_timestamp"], utc=True)
                pb = pb[(ts.dt.month == ent.month) & (ts.dt.year == ent.year)]
            if ent.on_date is not None:
                pb = pb[pb["game_date"] == ent.on_date]
        if not pb.empty:
            games = catalog.games[catalog.games["game_id"].isin(pb["game_id"])]

    boxes = filter_boxes(catalog, games, ent)
    recaps = catalog.recaps[catalog.recaps["game_id"].isin(games["game_id"])] if not games.empty else catalog.recaps.iloc[0:0]
    notes = catalog.notes[catalog.notes["game_id"].isin(games["game_id"])] if not games.empty else catalog.notes.iloc[0:0]
    if ent.player_ids:
        notes = catalog.notes[catalog.notes["player_id"].isin(ent.player_ids)]
        if not games.empty:
            notes = notes[notes["game_id"].isin(games["game_id"])]

    vector_rows: list[dict] = []
    declared = {s["table"] for s in return_spec.get("evidence", [])}
    need_vector = games.empty or (declared & {"game_recaps", "injury_notes"} and recaps.empty and notes.empty)
    if need_vector:
        qvec = ollama_embed(EMBED_MODEL, QUERY_PREFIX + question_text)
        if "game_details" in declared or not declared:
            for r in retrieve(cx, qvec, TOP_K):
                r = dict(r)
                r["table"] = "game_details"
                vector_rows.append(r)
            extra_ids = [int(r["game_id"]) for r in vector_rows]
            extra = catalog.games[catalog.games["game_id"].isin(extra_ids)]
            games = pd.concat([games, extra]).drop_duplicates("game_id") if not games.empty else extra
        if "player_box_scores" in declared:
            for r in retrieve_table(cx, "player_box_scores", ["game_id", "person_id"], qvec, TOP_K):
                r["table"] = "player_box_scores"
                vector_rows.append(r)
        if "game_recaps" in declared:
            for r in retrieve_table(cx, "game_recaps", ["recap_id", "game_id"], qvec, TOP_K):
                r["table"] = "game_recaps"
                r["id"] = r.get("recap_id")
                vector_rows.append(r)
            recap_ids = [int(r["recap_id"]) for r in vector_rows if r.get("table") == "game_recaps"]
            recaps = catalog.recaps[catalog.recaps["recap_id"].isin(recap_ids)]
            if games.empty:
                games = catalog.games[catalog.games["game_id"].isin(recaps["game_id"])]
        if "injury_notes" in declared:
            for r in retrieve_table(cx, "injury_notes", ["note_id", "game_id"], qvec, TOP_K):
                r["table"] = "injury_notes"
                r["id"] = r.get("note_id")
                vector_rows.append(r)
            note_ids = [int(r["note_id"]) for r in vector_rows if r.get("table") == "injury_notes"]
            notes = catalog.notes[catalog.notes["note_id"].isin(note_ids)]

    if recaps.empty and not games.empty:
        recaps = catalog.recaps[catalog.recaps["game_id"].isin(games["game_id"])]
    if notes.empty and not games.empty and ent.player_ids:
        notes = catalog.notes[
            catalog.notes["player_id"].isin(ent.player_ids) & catalog.notes["game_id"].isin(games["game_id"])
        ]
    return WorkingSet(games=games, boxes=boxes, recaps=recaps, notes=notes, vector_rows=vector_rows)


def answer_question(cx: Connection, question: dict, catalog: Catalog) -> dict:
    """Answer one question via retrieve -> reason -> format, driven by `return`."""
    return_spec = question["return"]
    ent = parse_entities(question["question"], catalog)
    ws = build_working_set(cx, question["question"], catalog, ent, return_spec)
    declared_shapes = return_spec.get("evidence", [])

    needs_boxes = any(s.get("table") == "player_box_scores" for s in declared_shapes)
    if needs_boxes and ent.player_ids and ws.boxes.empty and (ent.on_date is not None or ent.opener):
        return abstain_result(return_spec)

    derived = derive_from_tables(question["question"], return_spec, catalog, ws, ent)
    if derived is not None:
        derived = normalize_team_names(derived, catalog.teams)
        derived = normalize_player_names(derived, catalog)
        # Table-derived evidence is built from catalog rows; ground against the
        # full tables so a tighter working-set filter cannot drop valid cites.
        retrieved = retrieved_index(catalog.games, catalog.boxes, catalog.recaps, catalog.notes)
        grounded = ground_evidence(derived.get("evidence"), declared_shapes, retrieved)
        derived["evidence"] = grounded
        checked = verify_numeric(derived, catalog)
        if checked is not None and checked.get("answerable") and evidence_fully_grounded(declared_shapes, grounded):
            return {k: checked[k] for k in ["answerable", *[x for x in return_spec if x != "evidence"], "evidence"] if k in checked}

    ctx = compact_context(ws, catalog, ent)
    if not ctx.strip():
        return abstain_result(return_spec)
    parsed = llm_fill(question, ctx, return_spec, catalog.teams)
    if parsed is None:
        return abstain_result(return_spec)
    retrieved = retrieved_index(ws.games, ws.boxes, ws.recaps, ws.notes)
    grounded = ground_evidence(parsed.get("_raw_evidence"), declared_shapes, retrieved)
    result = {k: v for k, v in parsed.items() if k != "_raw_evidence"}
    result = normalize_player_names(result, catalog)
    if not result.get("answerable", False) or not evidence_fully_grounded(declared_shapes, grounded):
        return abstain_result(return_spec)
    result["evidence"] = grounded
    checked = verify_numeric(result, catalog)
    if checked is None:
        return abstain_result(return_spec)
    return checked


def format_chat_answer(result: dict) -> str:
    if not result.get("answerable"):
        return "I cannot answer that from the available data."
    parts = []
    skip = {"answerable", "evidence"}
    for key, value in result.items():
        if key in skip:
            continue
        parts.append(f"{key}: {value}")
    return "; ".join(parts) if parts else json.dumps({k: v for k, v in result.items() if k != "evidence"})


def answer_chat(cx: Connection, question_text: str, catalog: Catalog) -> dict:
    """Free-text chat path used by /api/chat — same retrieve/reason, prose answer."""
    dummy = {
        "question": question_text,
        "return": {
            "answerable": "bool",
            "answer": "str",
            "evidence": [
                {"table": "game_details", "id": "int"},
                {"table": "player_box_scores", "game_id": "int", "person_id": "int"},
                {"table": "game_recaps", "id": "int"},
                {"table": "injury_notes", "id": "int"},
            ],
        },
    }
    # Reuse entity retrieval; then generate a short grounded answer.
    ent = parse_entities(question_text, catalog)
    ws = build_working_set(cx, question_text, catalog, ent, dummy["return"])
    derived = derive_from_tables(
        question_text,
        {"answerable": "bool", "points": "int", "evidence": dummy["return"]["evidence"]},
        catalog,
        ws,
        ent,
    )
    ctx = compact_context(ws, catalog, ent)
    prompt = (
        "Answer the basketball question using only this context. Tables beat recaps. "
        "Be concise. If you cannot answer, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question_text}\nAnswer:"
    )
    if derived and derived.get("answerable"):
        answer = format_chat_answer(derived)
    elif ctx.strip():
        answer = ollama_generate(LLM_MODEL, prompt).strip()
    else:
        answer = "I cannot answer that from the available data."
    evidence = []
    for row in ws.games.head(8).itertuples(index=False):
        evidence.append({"table": "game_details", "id": int(row.game_id),
                         "doc": f"{row.game_date} {row.home_name} {int(row.home_points)}-{int(row.away_points)} {row.away_name}"})
    for row in ws.boxes.head(8).itertuples(index=False):
        evidence.append({"table": "player_box_scores", "game_id": int(row.game_id), "person_id": int(row.person_id),
                         "doc": f"{row.player_name} {int(row.points)} points"})
    for row in ws.recaps.head(4).itertuples(index=False):
        evidence.append({"table": "game_recaps", "id": int(row.recap_id), "doc": str(row.text)[:400]})
    for row in ws.notes.head(4).itertuples(index=False):
        evidence.append({"table": "injury_notes", "id": int(row.note_id), "doc": str(row.text)})
    return {"answer": answer, "evidence": evidence}


def main() -> None:
    """Answer every question in part1/questions.json and write part1/answers.json."""
    print("Starting RAG Answering")
    eng = sa.create_engine(DB_DSN)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    outputs = []
    with eng.begin() as cx:
        catalog = load_catalog(cx)
        for q in questions:
            result = answer_question(cx, q, catalog)
            outputs.append({"id": q["id"], "result": result})
            print(f"Q{q['id']}: {q['question']}\n  -> {json.dumps(result)}")

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Finished RAG Answering: wrote {len(outputs)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
