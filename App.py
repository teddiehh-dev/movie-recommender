# python -m streamlit run app.py
import asyncio
import json
import random
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 🔑 API KEYS (PULLED FROM STREAMLIT SECRETS)
# ==========================================
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    TMDB_API_KEY = 'YOUR_TMDB_API_KEY_HERE'
    GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY_HERE'

# ==========================================
# FIXED SETTINGS
# ==========================================
FIXED_TEMPERATURE = 1.2
MAX_CONCURRENCY = 10
NEUTRAL_RATING = 3.0
DISLIKE_CEILING = 2.0
MAX_HISTORY_BATCHES = 20

BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
REQUIRED_COLUMNS = {"Name", "Rating", "Year", "Date"}
CHARACTER_PHOTOS_DIR = Path(__file__).parent / "character_photos"

USER_DATA_DIR = Path(__file__).parent / "user_data"
USER_DATA_DIR.mkdir(exist_ok=True)


def get_watchlist_file():
    return USER_DATA_DIR / f"watchlist_{st.session_state.session_id}.json"


def get_history_file():
    return USER_DATA_DIR / f"history_{st.session_state.session_id}.json"

# ==========================================
# CHARACTER ARCHETYPES (from user-provided character_archetypes.csv)
# ==========================================
CHARACTER_ARCHETYPES = [
    {"character": "Ellen Ripley", "movie": "Alien", "traits": "Resilient survivor, no-nonsense pragmatist, and fierce protector."},
    {"character": "Indiana Jones", "movie": "Raiders of the Lost Ark", "traits": "Adventurous academic, improvisational risk-taker, and charming rogue."},
    {"character": "Amélie Poulain", "movie": "Amélie", "traits": "Whimsical introvert, quiet idealist, and imaginative romantic."},
    {"character": "Travis Bickle", "movie": "Taxi Driver", "traits": "Alienated loner, obsessive anti-hero, and gritty neo-noir cynic."},
    {"character": "The Dude", "movie": "The Big Lebowski", "traits": "Ultimate slacker, unbothered pacifist, and laid-back philosopher."},
    {"character": "Marge Gunderson", "movie": "Fargo", "traits": "Methodical detective, cheerful optimist, and grounded moral compass."},
    {"character": "Tony Stark", "movie": "Iron Man", "traits": "Brilliant egoist, quick-witted innovator, and complex futurist."},
    {"character": "Miles Morales", "movie": "Into the Spider-Verse", "traits": "Creative underdog, resilient coming-of-age hero, and vibrant idealist."},
    {"character": "Clarice Starling", "movie": "Silence of the Lambs", "traits": "Focused investigator, psychological explorer, and quiet underdog."},
    {"character": "Furiosa", "movie": "Mad Max: Fury Road", "traits": "Stoic rebel, high-octane action icon, and relentless liberator."},
    {"character": "Neo", "movie": "The Matrix", "traits": "Chosen savior, truth-seeker, and reality-bending cyberpunk hero."},
    {"character": "John Wick", "movie": "John Wick", "traits": "Legendary assassin, hyper-focused avenger, and hyper-stylized professional."},
    {"character": "Eowyn", "movie": "The Lord of the Rings", "traits": "Defiant warrior, noble underdog, and fierce rule-breaker."},
    {"character": "Tyler Durden", "movie": "Fight Club", "traits": "Nihilistic anarchist, chaotic anti-establishment rebel, and internal id."},
    {"character": "Vito Corleone", "movie": "The Godfather", "traits": "Calculated strategist, protective patriarch, and classical tragic authority."},
    {"character": "Leia Organa", "movie": "Star Wars", "traits": "Diplomatic leader, sharp-tongued revolutionary, and galactic royalty."},
    {"character": "Rick Blaine", "movie": "Casablanca", "traits": "Cynical ex-patriot, secret romantic, and classic noir tragic hero."},
    {"character": "Beatrix Kiddo", "movie": "Kill Bill", "traits": "Relentless weapon, vengeful mother, and stylized martial arts force."},
    {"character": "Ferris Bueller", "movie": "Ferris Bueller's Day Off", "traits": "Hyper-confident trickster, rule-bending hedonist, and charismatic host."},
    {"character": "Jules Winnfield", "movie": "Pulp Fiction", "traits": "Philosophical hitman, highly articulate talker, and repentant sinner."},
    {"character": "Paddington Bear", "movie": "Paddington", "traits": "Pure-hearted optimist, polite chaotic neutral, and wholesome center."},
    {"character": "Daniel Plainview", "movie": "There Will Blood", "traits": "Obsessive capitalist, misanthropic tycoon, and ruthless isolationist."},
    {"character": "Joel Barish", "movie": "Eternal Sunshine", "traits": "Melancholic dreamer, artistic introvert, and emotionally fragile romantic."},
    {"character": "Mary Poppins", "movie": "Mary Poppins", "traits": "Practically perfect authority, whimsical mentor, and magical guide."},
    {"character": "Ethan Hunt", "movie": "Mission: Impossible", "traits": "Adrenaline-fueled stuntman, unstoppable puzzle-solver, and teammate."},
    {"character": "Hermione Granger", "movie": "Harry Potter", "traits": "Book-smart tactician, fiercely loyal defender, and moral stickler."},
    {"character": "Marty McFly", "movie": "Back to the Future", "traits": "Everyman time-traveler, cool-kid adaptivist, and reactive hero."},
    {"character": "Sarah Connor", "movie": "Terminator 2", "traits": "Hardened survivalist, fiercely protective matriarch, and doomsday prophet."},
    {"character": "Atticus Finch", "movie": "To Kill a Mockingbird", "traits": "Unshakable moralist, dignified defender, and classic legal patriarch."},
    {"character": "James Bond", "movie": "Casino Royale", "traits": "Smooth operator, lethal government asset, and high-society sybarite."},
]

# ==========================================
# SMALL HELPERS
# ==========================================
def year_to_str(year):
    if pd.notna(year):
        try:
            return str(int(year))
        except (ValueError, TypeError):
            return ""
    return ""


def normalize_title(title):
    return (title or "").strip().lower()


def safe_filename(name):
    keep = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in (name or ""))
    return keep.strip().replace(" ", "_")


def get_local_character_photo(character_name):
    if not CHARACTER_PHOTOS_DIR.exists():
        return None
    stem = safe_filename(character_name)
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = CHARACTER_PHOTOS_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_trailer_url(videos):
    yt_videos = [v for v in videos if v.get("site") == "YouTube"]
    trailers = [v for v in yt_videos if v.get("type") == "Trailer"]
    official_trailers = [v for v in trailers if v.get("official")]

    pick = None
    if official_trailers:
        pick = official_trailers[0]
    elif trailers:
        pick = trailers[0]
    elif yt_videos:
        pick = yt_videos[0]

    if pick:
        return f"https://www.youtube.com/watch?v={pick['key']}"
    return None


def normalize_provider_name(name):
    lname = (name or "").lower()
    if "amazon" in lname and "prime" in lname:
        return "Amazon Prime Video"
    return name


def extract_streaming_for_region(watch_providers_by_region, region):
    if not watch_providers_by_region:
        return None
    region_data = watch_providers_by_region.get(region)
    if not region_data:
        return None

    def names(key):
        raw = [p["provider_name"] for p in region_data.get(key, [])]
        grouped, seen = [], set()
        for n in raw:
            norm = normalize_provider_name(n)
            if norm not in seen:
                seen.add(norm)
                grouped.append(norm)
        return grouped

    flatrate, rent, buy = names("flatrate"), names("rent"), names("buy")
    if not (flatrate or rent or buy):
        return None
    return {"flatrate": flatrate, "rent": rent, "buy": buy, "link": region_data.get("link")}


def inject_custom_css():
    st.markdown(
        """
        <style>
        .badge {
            display: inline-block;
            background-color: #F2D7D9;
            color: #6B3F3F;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            margin: 2px 4px 2px 0;
            font-family: serif;
        }
        .badge-streaming { background-color: #A8C3B7; color: #234D3B; }
        .badge-rent { background-color: #F4D35E; color: #6B4226; }
        .fade-img {
            width: 100%;
            max-width: 220px;
            display: block;
            border-radius: 6px;
            border: 1px solid #D9C7A3;
            animation: fadeInPoster 0.5s ease;
        }
        @keyframes fadeInPoster {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_poster_html(url, alt="poster"):
    st.markdown(
        f'<img src="{url}" alt="{alt}" loading="lazy" class="fade-img" />',
        unsafe_allow_html=True,
    )


def render_badges(items, kind="genre"):
    if not items:
        return
    css_class = "badge" if kind == "genre" else f"badge {kind}"
    spans = "".join(f'<span class="{css_class}">{item}</span>' for item in items)
    st.markdown(spans, unsafe_allow_html=True)


def render_streaming_line(streaming):
    if not streaming:
        st.caption("📺 No streaming info for this region")
        return
    if streaming.get("flatrate"):
        st.caption("📺 Streaming")
        render_badges(streaming["flatrate"], kind="badge-streaming")
    if streaming.get("rent") or streaming.get("buy"):
        rent_buy = streaming.get("rent", []) + streaming.get("buy", [])
        st.caption("💰 Rent/Buy")
        render_badges(rent_buy, kind="badge-rent")


# ==========================================
# WATCHLIST PERSISTENCE
# ==========================================
def load_watchlist():
    watchlist_file = get_watchlist_file()
    if watchlist_file.exists():
        try:
            with open(watchlist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_watchlist(watchlist):
    try:
        with open(get_watchlist_file(), "w", encoding="utf-8") as f:
            json.dump(watchlist, f, indent=2)
    except OSError as e:
        st.error(f"Couldn't save watchlist to disk: {e}")


def add_to_watchlist(rec):
    key = normalize_title(rec.get("title"))
    existing = {normalize_title(w["title"]) for w in st.session_state.watchlist}
    if key not in existing:
        st.session_state.watchlist.append({
            "title": rec.get("title"),
            "year": rec.get("year"),
            "director": rec.get("director"),
            "poster_path": rec.get("poster_path"),
        })
        save_watchlist(st.session_state.watchlist)


def remove_from_watchlist(title):
    key = normalize_title(title)
    st.session_state.watchlist = [
        w for w in st.session_state.watchlist if normalize_title(w["title"]) != key
    ]
    save_watchlist(st.session_state.watchlist)


# ==========================================
# RECOMMENDATION HISTORY PERSISTENCE
# ==========================================
def load_history():
    history_file = get_history_file()
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(history):
    try:
        with open(get_history_file(), "w", encoding="utf-8") as f:
            json.dump(history[-MAX_HISTORY_BATCHES:], f, indent=2)
    except OSError as e:
        st.error(f"Couldn't save recommendation history: {e}")


def add_history_batch(recommendations, streaming_region):
    snapshot = []
    for rec in recommendations:
        snapshot.append({
            "title": rec.get("title"),
            "year": rec.get("year"),
            "director": rec.get("director"),
            "match_type": rec.get("match_type"),
            "reason": rec.get("reason"),
            "poster_path": rec.get("poster_path"),
            "vote_average": rec.get("vote_average"),
            "trailer_url": rec.get("trailer_url"),
            "streaming": extract_streaming_for_region(rec.get("watch_providers"), streaming_region),
        })
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "recommendations": snapshot,
    })
    save_history(st.session_state.history)


# ==========================================
# ASYNC TMDB CORE
# ==========================================
async def search_movie_id_async(client, title, year_str, sem, retries=2):
    search_params = {"api_key": TMDB_API_KEY, "query": title, "year": year_str}
    async with sem:
        for attempt in range(retries + 1):
            try:
                resp = await client.get(f"{BASE_URL}/search/movie", params=search_params, timeout=8)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 1))
                    await asyncio.sleep(min(wait, 5))
                    continue
                if resp.status_code == 401:
                    return None, "auth_error"
                resp.raise_for_status()
                payload = resp.json()
                if not payload.get("results"):
                    return None, "no_match"
                return payload["results"][0]["id"], "ok"
            except httpx.TimeoutException:
                if attempt == retries:
                    return None, "timeout"
            except httpx.HTTPError:
                if attempt == retries:
                    return None, "network_error"
    return None, "network_error"


async def fetch_movie_details_by_id_async(client, movie_id, sem=None, retries=2):
    details_params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,videos,watch/providers,images",
        "include_image_language": "en,null",
    }

    async def _do():
        for attempt in range(retries + 1):
            try:
                resp = await client.get(f"{BASE_URL}/movie/{movie_id}", params=details_params, timeout=8)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 1))
                    await asyncio.sleep(min(wait, 5))
                    continue
                resp.raise_for_status()
                details = resp.json()
                backdrops = details.get("images", {}).get("backdrops", [])
                return {
                    "id": movie_id,
                    "title": details.get("title"),
                    "overview": details.get("overview"),
                    "year": (details.get("release_date") or "")[:4],
                    "runtime": details.get("runtime"),
                    "genres": [g["name"] for g in details.get("genres", [])],
                    "directors": [c["name"] for c in details.get("credits", {}).get("crew", []) if c["job"] == "Director"],
                    "actors": [c["name"] for c in details.get("credits", {}).get("cast", [])[:5]],
                    "cast_raw": [
                        {"name": c.get("name"), "character": c.get("character", ""), "profile_path": c.get("profile_path")}
                        for c in details.get("credits", {}).get("cast", [])[:20]
                    ],
                    "backdrop_path": backdrops[0]["file_path"] if backdrops else None,
                    "poster_path": details.get("poster_path"),
                    "vote_average": details.get("vote_average"),
                    "vote_count": details.get("vote_count"),
                    "trailer_url": extract_trailer_url(details.get("videos", {}).get("results", [])),
                    "watch_providers": details.get("watch/providers", {}).get("results", {}),
                    "origin_country": details.get("origin_country", []), # NEW - used for fun facts
                }
            except httpx.HTTPError:
                if attempt == retries:
                    return None
        return None

    if sem:
        async with sem:
            return await _do()
    return await _do()


async def fetch_single_movie_async(client, title, year, sem, retries=2):
    year_str = year if isinstance(year, str) else year_to_str(year)
    movie_id, status = await search_movie_id_async(client, title, year_str, sem, retries)
    if not movie_id:
        return None, status
    details = await fetch_movie_details_by_id_async(client, movie_id, sem, retries)
    if not details:
        return None, "network_error"
    return details, "ok"


# ==========================================
# GENERIC PROGRESS-TRACKED BATCH FETCH
# ==========================================
async def _fetch_movies_with_progress(title_year_list, max_concurrency, progress_bar=None):
    cache = st.session_state.tmdb_cache

    seen_keys = set()
    to_fetch = []
    for title, year in title_year_list:
        key = (normalize_title(title), year)
        if key not in cache and key not in seen_keys:
            to_fetch.append((title, year))
            seen_keys.add(key)

    total = len(title_year_list)
    already = total - len(to_fetch)
    if progress_bar:
        progress_bar.progress(already / total if total else 1.0)

    if to_fetch:
        sem = asyncio.Semaphore(max_concurrency)
        limits = httpx.Limits(max_connections=max_concurrency * 2, max_keepalive_connections=max_concurrency)
        completed = already

        async with httpx.AsyncClient(limits=limits) as client:
            async def wrapped(title, year):
                data, status = await fetch_single_movie_async(client, title, year, sem)
                return title, year, data, status

            tasks = [asyncio.create_task(wrapped(t, y)) for t, y in to_fetch]
            for coro in asyncio.as_completed(tasks):
                title, year, data, status = await coro
                cache[(normalize_title(title), year)] = (data, status)
                completed += 1
                if progress_bar:
                    progress_bar.progress(completed / total)


def fetch_analysis_movies(movies_with_rating, max_concurrency, progress_bar=None):
    title_year_list = [(t, y) for t, y, _ in movies_with_rating]
    asyncio.run(_fetch_movies_with_progress(title_year_list, max_concurrency, progress_bar))
    cache = st.session_state.tmdb_cache
    return [
        (t, y, r) + cache[(normalize_title(t), y)]
        for t, y, r in movies_with_rating
    ]


def fetch_recommendation_details(recs, max_concurrency, progress_bar=None):
    title_year_list = [(r["title"], r.get("year", "")) for r in recs]
    asyncio.run(_fetch_movies_with_progress(title_year_list, max_concurrency, progress_bar))
    cache = st.session_state.tmdb_cache
    for rec in recs:
        data, status = cache.get((normalize_title(rec["title"]), rec.get("year", "")), (None, "no_match"))
        if data:
            rec["poster_path"] = data.get("poster_path")
            rec["backdrop_path"] = data.get("backdrop_path")
            rec["overview"] = data.get("overview")
            rec["vote_average"] = data.get("vote_average")
            rec["vote_count"] = data.get("vote_count")
            rec["trailer_url"] = data.get("trailer_url")
            rec["genres"] = data.get("genres")
            rec["actors"] = data.get("actors")
            rec["cast_raw"] = data.get("cast_raw")
            rec["watch_providers"] = data.get("watch_providers")
            if not rec.get("director"):
                rec["director"] = ", ".join(data.get("directors", [])) or "Unknown"
            rec["verified_on_tmdb"] = True
        else:
            rec["poster_path"] = None
            rec["backdrop_path"] = None
            rec["overview"] = None
            rec["vote_average"] = None
            rec["vote_count"] = None
            rec["trailer_url"] = None
            rec["genres"] = None
            rec["actors"] = None
            rec["cast_raw"] = None
            rec["watch_providers"] = None
            rec["verified_on_tmdb"] = False
    return recs


# ==========================================
# TRENDING THIS WEEK
# ==========================================
async def fetch_trending_async(max_concurrency):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/trending/movie/week", params={"api_key": TMDB_API_KEY}, timeout=8)
        resp.raise_for_status()
        top5 = resp.json().get("results", [])[:5]

        sem = asyncio.Semaphore(max_concurrency)
        detail_tasks = [fetch_movie_details_by_id_async(client, m["id"], sem) for m in top5]
        details_list = await asyncio.gather(*detail_tasks)

        for movie, details in zip(top5, details_list):
            movie["year"] = (movie.get("release_date") or "")[:4]
            if details:
                movie["trailer_url"] = details.get("trailer_url")
                movie["watch_providers"] = details.get("watch_providers")
                movie["director"] = ", ".join(details.get("directors", [])) or "Unknown"
                movie["actors"] = details.get("actors")
            else:
                movie["trailer_url"] = None
                movie["watch_providers"] = None
                movie["director"] = "Unknown"
                movie["actors"] = None
        return top5


@st.cache_data(ttl=3600, show_spinner=False)
def get_trending_movies(_api_key, max_concurrency):
    return asyncio.run(fetch_trending_async(max_concurrency))


# ==========================================
# RANDOM MOVIE
# ==========================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_genre_list(_api_key):
    resp = httpx.get(f"{BASE_URL}/genre/movie/list", params={"api_key": TMDB_API_KEY}, timeout=8)
    resp.raise_for_status()
    return resp.json().get("genres", [])


@st.cache_data(ttl=86400, show_spinner=False)
def get_watch_providers_list(_api_key, region):
    resp = httpx.get(
        f"{BASE_URL}/watch/providers/movie", params={"api_key": TMDB_API_KEY, "watch_region": region}, timeout=8
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


MIN_VOTE_COUNT = 500

async def discover_random_movie_async(genre_ids, provider_ids, region, max_concurrency, attempts=5):
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(max_concurrency)

        for _ in range(attempts):
            params = {
                "api_key": TMDB_API_KEY,
                "include_adult": "false",
                "include_video": "false",
                "vote_count.gte": MIN_VOTE_COUNT,
                "sort_by": "popularity.desc",
                "page": 1,
            }
            if genre_ids:
                params["with_genres"] = str(random.choice(genre_ids))
            if provider_ids:
                params["with_watch_providers"] = "|".join(str(p) for p in provider_ids)
                params["watch_region"] = region

            resp = await client.get(f"{BASE_URL}/discover/movie", params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            total_pages = data.get("total_pages", 0)

            if total_pages == 0:
                continue

            page = random.randint(1, min(total_pages, 500))
            if page != 1:
                params["page"] = page
                resp = await client.get(f"{BASE_URL}/discover/movie", params=params, timeout=8)
                resp.raise_for_status()
                data = resp.json()

            results = [
                r for r in data.get("results", [])
                if r.get("poster_path") and not r.get("video")
                and (r.get("vote_count") or 0) >= MIN_VOTE_COUNT
            ]

            if results:
                chosen = random.choice(results)
                details = await fetch_movie_details_by_id_async(client, chosen["id"], sem)
                if details:
                    return "ok", details

        return "no_results", None


# ==========================================
# GEMINI AI FUNCTIONS
# ==========================================
def find_similar_movies_gemini(title, year_str, temperature):
    year_bit = f" ({year_str})" if year_str else ""
    prompt = f"""
    Recommend exactly 5 movies that are genuinely similar to "{title}"{year_bit} in tone, themes,
    genre, pacing, or style.
    Do not include "{title}" itself in the list.
    Respond with ONLY a JSON array, no markdown code fences, no commentary, in this exact shape:
    [
      {{"title": "string", "year": "string (4-digit year)", "director": "string", "reason": "1-2 sentence explanation of the similarity"}}
    ]
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    return json.loads(raw_text)


def find_character_match_gemini(taste_profile, temperature):
    character_lines = "\n".join(
        f'{i+1}. {c["character"]} ({c["movie"]}) — {c["traits"]}'
        for i, c in enumerate(CHARACTER_ARCHETYPES)
    )

    avoid_bits = []
    if taste_profile.get("avoid_genres"):
        avoid_bits.append(f"genres: {', '.join(taste_profile['avoid_genres'])}")
    if taste_profile.get("avoid_directors"):
        avoid_bits.append(f"directors: {', '.join(taste_profile['avoid_directors'])}")
    avoid_text = f"\n    Tends to dislike: {'; '.join(avoid_bits)}" if avoid_bits else ""

    prompt = f"""
    Here is a film taste profile built from someone's Letterboxd ratings:
    - Core Genres: {', '.join(taste_profile.get('genres', []))}
    - Preferred Directors: {', '.join(taste_profile.get('directors', []))}
    - Key Actors: {', '.join(taste_profile.get('actors', []))}{avoid_text}

    From the following fixed list of film characters and their core traits, choose the ONE character
    whose personality and archetype best resonates with this taste profile. You MUST pick exactly one character from this list:

    {character_lines}

    Respond with ONLY a JSON object, no markdown code fences, no commentary, in this exact shape:
    {{"character": "string (exact name)", "movie": "string (exact movie)", "explanation": "~50 word explanation"}}
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    result = json.loads(raw_text)

    matched = next(
        (c for c in CHARACTER_ARCHETYPES if normalize_title(c["character"]) == normalize_title(result.get("character"))),
        None,
    )
    if matched:
        result["character"] = matched["character"]
        result["movie"] = matched["movie"]
        result["traits"] = matched["traits"]
        result["verified"] = True
    else:
        result["traits"] = None
        result["verified"] = False

    return result


def regenerate_recommendation_gemini(taste_profile, match_type, excluded_titles, temperature):
    avoid_bits = []
    if taste_profile.get("avoid_genres"):
        avoid_bits.append(f"genres: {', '.join(taste_profile['avoid_genres'])}")
    if taste_profile.get("avoid_directors"):
        avoid_bits.append(f"directors: {', '.join(taste_profile['avoid_directors'])}")
    avoid_text = f"\n    Tends to dislike: {'; '.join(avoid_bits)}" if avoid_bits else ""

    match_desc = (
        "a close, safe match to my genres, directors, and actors below"
        if match_type == "close_match"
        else (
            "an adventurous pick that stretches outside my usual profile but I'd still plausibly "
            "enjoy — explain the creative leap in the reason"
        )
    )

    prompt = f"""
    Here is my film taste profile:
    - Core Genres: {', '.join(taste_profile.get('genres', []))}
    - Preferred Directors: {', '.join(taste_profile.get('directors', []))}
    - Key Actors: {', '.join(taste_profile.get('actors', []))}{avoid_text}

    Recommend exactly 1 movie that is {match_desc}.

    CRITICAL CONSTRAINT: do not recommend any of the following:
    {', '.join(excluded_titles)}

    Write a "reason" of approximately 100 words explaining why this movie fits my taste profile.

    Respond with ONLY a JSON object, no markdown code fences, no commentary, in this exact shape:
    {{"title": "string", "year": "string (4-digit year)", "director": "string", "match_type": "{match_type}", "reason": "~100 word explanation"}}
    """

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    return json.loads(raw_text)


def refine_recommendation(rejected_rec):
    with st.spinner("Finding a replacement..."):
        try:
            current_titles = [r.get("title") for r in st.session_state.recommendations]
            seen = st.session_state.get("seen_movies") or []
            watchlist_titles = [w["title"] for w in st.session_state.watchlist]
            excluded = list(set(seen + watchlist_titles + current_titles))

            new_rec = regenerate_recommendation_gemini(
                st.session_state.taste_profile, rejected_rec.get("match_type"), excluded, FIXED_TEMPERATURE
            )
            new_rec = fetch_recommendation_details([new_rec], MAX_CONCURRENCY)[0]

            idx = next(
                (i for i, r in enumerate(st.session_state.recommendations) if r is rejected_rec), None
            )
            if idx is not None:
                st.session_state.recommendations[idx] = new_rec
            else:
                st.session_state.recommendations.append(new_rec)
        except json.JSONDecodeError:
            st.warning("Couldn't parse a replacement — try again.")
        except Exception as e:
            st.warning(f"Couldn't find a replacement: {e}")


def render_recommendation_card(key_prefix, rec, watchlist_keys, streaming_region, allow_refine=False):
    with st.container(border=True):
        poster_col, info_col = st.columns([1, 3])

        with poster_col:
            if rec.get("poster_path"):
                render_poster_html(f"{POSTER_BASE}{rec['poster_path']}", alt=rec.get("title", "poster"))
            else:
                st.caption("No poster found")

        with info_col:
            st.markdown(f"**{rec.get('title', 'Unknown')}** ({rec.get('year', '—')})")
            st.caption(f"Directed by {rec.get('director', 'Unknown')}")

            if rec.get("genres"):
                render_badges(rec["genres"], kind="genre")

            if rec.get("actors"):
                st.caption("Starring: " + ", ".join(rec["actors"]))

            if rec.get("vote_average") is not None:
                st.write(f"⭐ TMDB rating: {rec['vote_average']:.1f}/10 ({rec.get('vote_count', 0)} votes)")
            else:
                st.caption("TMDB rating unavailable")

            if rec.get("trailer_url"):
                st.markdown(f"[🎬 Watch Trailer]({rec['trailer_url']})")
            else:
                st.caption("No trailer found")

            if rec.get("overview"):
                st.markdown("**Description**")
                st.write(rec["overview"])

            with st.expander("More details"):
                if rec.get("verified_on_tmdb") is False:
                    st.warning("⚠️ Couldn't verify this title on TMDB — it may be inaccurate.")

                render_streaming_line(extract_streaming_for_region(rec.get("watch_providers"), streaming_region))

                if rec.get("reason"):
                    st.markdown("**Why this movie was recommended for you**")
                    st.write(rec["reason"])

            button_col1, button_col2 = st.columns([1, 1])
            with button_col1:
                is_on_watchlist = normalize_title(rec.get("title")) in watchlist_keys
                if is_on_watchlist:
                    st.success("✅ On your watchlist")
                else:
                    if st.button("➕ Add to Watchlist", key=f"add_{key_prefix}_{normalize_title(rec.get('title'))}"):
                        add_to_watchlist(rec)
                        st.rerun()

            with button_col2:
                if allow_refine and rec.get("match_type") in ("close_match", "adventurous_pick"):
                    if st.button("👎 Not for me — swap it", key=f"refine_{key_prefix}_{normalize_title(rec.get('title'))}"):
                        refine_recommendation(rec)
                        st.rerun()


def render_recommendation_grid(items, key_prefix, watchlist_keys, streaming_region, allow_refine=False):
    for idx, item in enumerate(items):
        render_recommendation_card(
            f"{key_prefix}_{idx}", item, watchlist_keys, streaming_region, allow_refine=allow_refine
        )


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Your Cinematic DNA", layout="wide")
inject_custom_css()
st.title("🧬 Your Cinematic DNA")
st.write(
    "Turn your Letterboxd history into actionable insights. Analyze your lifetime watch stats, "
    "visualize your unique taste profile, and receive AI-curated movie recommendations based on "
    "the genres, directors, and actors you love most."
)

with st.expander("ℹ️ About & Instructions: How to get your Letterboxd Data", expanded=False):
    st.markdown("""
    **How does this app work?**
    This application analyzes your entire Letterboxd watching history using the TMDB API. It calculates your lifetime stats (Fun Facts) and then feeds your highest and lowest rated movies to Google's Gemini AI to build a "Taste Profile" consisting of your favorite actors, directors, and genres. It then uses this profile to generate personalized movie recommendations.

    **How to get your `diary.csv` file from Letterboxd:**
    1. Log into your [Letterboxd](https://letterboxd.com) account on a desktop or mobile browser.
    2. Click your username in the top navigation bar and select **Settings**.
    3. Click on the **Import & Export** tab.
    4. Click the **Export your data** button. 
    5. This will download a `.zip` file to your computer. Unzip it.
    6. Inside that folder, you will find a file named **`diary.csv`**. 
    7. Upload that exact file into the uploader below!
    """)

st.sidebar.header("🔧 Settings")
rating_threshold = st.sidebar.slider(
    "Treat movies rated at or above this as 'liked':",
    min_value=1.0, max_value=5.0, value=3.5, step=0.5
)
streaming_region = "GB" 
st.sidebar.caption("Streaming region: UK")
sampling_strategy = st.sidebar.radio(
    "Which movies to analyze",
    ["Most/least rated (recommended)", "Random sample"],
    help=(
        "Most/least rated uses your strongest opinions (highest and lowest ratings) for the "
        "clearest taste signal. Random sample pulls a random subset from your liked/disliked "
        "pools instead — weaker signal per movie, but gives more variety across runs."
    ),
)
st.sidebar.caption(f"Temperature fixed at {FIXED_TEMPERATURE}. TMDB concurrency fixed at {MAX_CONCURRENCY}.")
st.sidebar.caption("Your watchlist and history are private to this session and aren't saved permanently.")

# ---- session state init ----
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "taste_profile" not in st.session_state:
    st.session_state.taste_profile = None
if "fun_facts" not in st.session_state:
    st.session_state.fun_facts = None
if "monthly_watch_data" not in st.session_state:
    st.session_state.monthly_watch_data = None
if "character_match" not in st.session_state:
    st.session_state.character_match = None
if "tmdb_cache" not in st.session_state:
    st.session_state.tmdb_cache = {}
if "more_like_this_results" not in st.session_state:
    st.session_state.more_like_this_results = None
if "random_movie" not in st.session_state:
    st.session_state.random_movie = None
if "random_movie_provider_filter" not in st.session_state:
    st.session_state.random_movie_provider_filter = set()
if "seen_movies" not in st.session_state:
    st.session_state.seen_movies = []


def render_recommendations_tab():
    # ---- trending ----
    st.subheader("🔥 Trending This Week")
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        st.caption("Add your TMDB API key at the top of the script to see trending movies.")
    else:
        try:
            trending = get_trending_movies(TMDB_API_KEY, MAX_CONCURRENCY)
            watchlist_keys = {normalize_title(w["title"]) for w in st.session_state.watchlist}
            trend_cols = st.columns(5)
            for col, movie in zip(trend_cols, trending):
                with col:
                    with st.container(border=True):
                        if movie.get("poster_path"):
                            render_poster_html(f"{POSTER_BASE}{movie['poster_path']}", alt=movie.get("title", "poster"))
                        st.caption(f"**{movie.get('title', 'Unknown')}**")
                        if movie.get("vote_average") is not None:
                            st.caption(f"⭐ {movie['vote_average']:.1f}/10")
                        if movie.get("trailer_url"):
                            st.markdown(f"[🎬 Trailer]({movie['trailer_url']})")
                        streaming = extract_streaming_for_region(movie.get("watch_providers"), streaming_region)
                        if streaming and streaming.get("flatrate"):
                            render_badges(streaming["flatrate"][:2], kind="badge-streaming")

                        with st.expander("More details"):
                            st.caption(f"Directed by {movie.get('director', 'Unknown')}")
                            if movie.get("actors"):
                                st.caption("Starring: " + ", ".join(movie["actors"]))
                            if movie.get("overview"):
                                st.write(movie["overview"])

                        is_on_watchlist = normalize_title(movie.get("title")) in watchlist_keys
                        if is_on_watchlist:
                            st.caption("✅ On watchlist")
                        else:
                            if st.button("➕ Watchlist", key=f"trend_add_{movie.get('id')}"):
                                add_to_watchlist({
                                    "title": movie.get("title"),
                                    "year": movie.get("year"),
                                    "director": movie.get("director"),
                                    "poster_path": movie.get("poster_path"),
                                })
                                st.rerun()
        except httpx.HTTPError as e:
            st.caption(f"Couldn't load trending movies right now ({e}).")

    st.write("---")

    uploaded_file = st.file_uploader("Upload your Letterboxd 'diary.csv' file", type=["csv"])

    # ==========================================
    # MAIN EXECUTION PIPELINE
    # ==========================================
    if uploaded_file is not None:
        if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE" or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            st.error("⚠️ You need to set your API keys either in Streamlit secrets or directly in the code!")
        else:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Couldn't read that CSV: {e}")
                st.stop()

            missing_cols = REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                st.error(
                    f"This doesn't look like a Letterboxd `diary.csv` — missing column(s): "
                    f"{', '.join(sorted(missing_cols))}."
                )
                st.stop()
            
            # --- NEW DIARY.CSV CLEANING LOGIC ---
            
            # 1. Drop any diary entries where you didn't leave a star rating
            df = df.dropna(subset=["Rating"])
            
            # 2. Drop duplicate movie entries (rewatches) so they don't skew the AI profile.
            # keep="last" ensures it keeps your most recent rating for that film.
            df = df.drop_duplicates(subset=["Name", "Year"], keep="last")
            
            # ------------------------------------

            # Prepare Line Graph Data
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Month_Year'] = df['Date'].dt.to_period('M')
                monthly_counts = df.groupby('Month_Year').size().reset_index(name='Movies Watched')
                monthly_counts['Month_Year'] = monthly_counts['Month_Year'].dt.to_timestamp()
                st.session_state.monthly_watch_data = monthly_counts
            except Exception as e:
                 st.caption(f"Could not parse 'Date' column for line graph: {e}")
                 st.session_state.monthly_watch_data = None

            seen_movies = df["Name"].dropna().tolist()
            st.session_state.seen_movies = seen_movies
            liked_movies = df[df["Rating"] >= rating_threshold].copy()
            disliked_movies = df[df["Rating"] <= DISLIKE_CEILING].copy()

            st.write(
                f"Found **{len(liked_movies)}** liked movies (≥{rating_threshold}★) "
                f"and **{len(disliked_movies)}** disliked movies (≤{DISLIKE_CEILING}★) to build your taste profile from."
            )

            if len(liked_movies) == 0:
                st.warning("No movies found matching this threshold. Try lowering the star rating in the sidebar.")
            else:
                if st.button("⚡ Generate Fast Recommendations & Fun Facts"):
                    start_time = time.time()

                    # -------------------------------------------------------------
                    # 1. FETCH & ANALYZE 100% OF CSV FILMS FOR EXACT STATS
                    # -------------------------------------------------------------
                    st.caption(f"Fetching full metadata for all {len(df)} movies in your CSV from TMDB...")
                    all_csv_title_years = [
                        (row["Name"], year_to_str(row["Year"])) for _, row in df.iterrows()
                    ]
                    
                    full_progress_bar = st.progress(0.0)
                    asyncio.run(_fetch_movies_with_progress(all_csv_title_years, MAX_CONCURRENCY, full_progress_bar))
                    
                    analyzed_full_data = []
                    for _, row in df.iterrows():
                        t_norm = normalize_title(row["Name"])
                        y_str = year_to_str(row["Year"])
                        user_rating = float(row["Rating"]) if pd.notna(row["Rating"]) else None
                        data, status = st.session_state.tmdb_cache.get((t_norm, y_str), (None, "no_match"))
                        
                        if data and status == "ok":
                            yr_parsed = None
                            if data.get("year") and str(data.get("year")).isdigit():
                                yr_parsed = int(data.get("year"))
                            
                            analyzed_full_data.append({
                                "Title": data.get("title") or row["Name"],
                                "User_Rating": user_rating,
                                "TMDB_Rating": data.get("vote_average"),
                                "TMDB_Votes": data.get("vote_count"),
                                "Runtime": data.get("runtime"),
                                "Year": yr_parsed,
                                "Directors": data.get("directors", []),
                                "Actors": data.get("actors", []),
                                "Countries": data.get("origin_country", [])
                            })

                    # --- CALCULATE FUN FACTS ---
                    ff = {}
                    
                    # Watch Counts
                    ff["total_movies"] = len(analyzed_full_data)
                    all_actors = Counter()
                    all_directors = Counter()
                    all_countries = Counter()
                    
                    for d in analyzed_full_data:
                        for a in d["Actors"]: all_actors[a] += 1
                        for dr in d["Directors"]: all_directors[dr] += 1
                        for c in d["Countries"]: all_countries[c] += 1
                        
                    ff["most_watched_actor"] = all_actors.most_common(1)[0] if all_actors else None
                    ff["most_watched_director"] = all_directors.most_common(1)[0] if all_directors else None
                    ff["top_countries"] = all_countries.most_common(3) if all_countries else None
                    
                    movie_counts = df["Name"].value_counts()
                    ff["most_logged_movie"] = (movie_counts.index[0], movie_counts.iloc[0]) if not df.empty else None

                    # Ratings Quirks
                    valid_ratings = [d for d in analyzed_full_data if d["User_Rating"] is not None and d["TMDB_Rating"] is not None]
                    if valid_ratings:
                        for d in valid_ratings:
                            d["Diff"] = (d["User_Rating"] * 2) - d["TMDB_Rating"]
                        
                        valid_ratings.sort(key=lambda x: x["Diff"])
                        ff["biggest_hot_take"] = valid_ratings[0]  
                        ff["biggest_hidden_gem"] = valid_ratings[-1] 
                        
                        user_ratings_only = [d["User_Rating"] for d in valid_ratings]
                        ff["mean_rating"] = sum(user_ratings_only) / len(user_ratings_only)
                        ff["most_common_rating"] = Counter(user_ratings_only).most_common(1)[0][0]

                    # Watch Time & Length
                    valid_runtimes = [d for d in analyzed_full_data if d["Runtime"]]
                    if valid_runtimes:
                        total_mins = sum(d["Runtime"] for d in valid_runtimes)
                        ff["total_watch_days"] = total_mins // (24 * 60)
                        ff["total_watch_hours"] = (total_mins % (24 * 60)) // 60
                        ff["avg_runtime"] = total_mins // len(valid_runtimes)
                        ff["longest_movie"] = max(valid_runtimes, key=lambda x: x["Runtime"])
                        ff["shortest_movie"] = min(valid_runtimes, key=lambda x: x["Runtime"])

                    # Release Era & Chronology
                    valid_years = [d for d in analyzed_full_data if d["Year"]]
                    if valid_years:
                        ff["oldest_movie"] = min(valid_years, key=lambda x: x["Year"])
                        ff["newest_movie"] = max(valid_years, key=lambda x: x["Year"])
                        
                        decades = [(d["Year"] // 10) * 10 for d in valid_years]
                        ff["most_watched_decade"] = Counter(decades).most_common(1)[0][0]
                        
                        decade_scores = {}
                        for d in valid_years:
                            if d["User_Rating"] is not None:
                                dec = (d["Year"] // 10) * 10
                                decade_scores.setdefault(dec, []).append(d["User_Rating"])
                        
                        if decade_scores:
                            ff["highest_rated_decade"] = max(decade_scores.keys(), key=lambda k: sum(decade_scores[k])/len(decade_scores[k]))

                    # Niche & Underground
                    valid_votes = [d for d in analyzed_full_data if d["TMDB_Votes"] is not None and d["TMDB_Votes"] > 0]
                    if valid_votes:
                        ff["deepest_cut"] = min(valid_votes, key=lambda x: x["TMDB_Votes"])
                        
                        community_hits = [d for d in valid_votes if d["TMDB_Votes"] >= 100]
                        if community_hits:
                            ff["highest_community"] = max(community_hits, key=lambda x: x["TMDB_Rating"])

                    st.session_state.fun_facts = ff

                    # -------------------------------------------------------------
                    # 2. SAMPLE FOR GEMINI AI TASTE PROFILE & RECOMMENDATIONS
                    # -------------------------------------------------------------
                    if sampling_strategy == "Random sample":
                        liked_sample = liked_movies.sample(n=min(45, len(liked_movies)), random_state=None)
                        disliked_sample = disliked_movies.sample(n=min(15, len(disliked_movies)), random_state=None)
                    else:
                        liked_sample = liked_movies.sort_values("Rating", ascending=False).head(45)
                        disliked_sample = disliked_movies.sort_values("Rating", ascending=True).head(15)
                    movies_to_analyze = pd.concat([liked_sample, disliked_sample], ignore_index=True)

                    movies_with_rating = [
                        (row["Name"], year_to_str(row["Year"]), float(row["Rating"]))
                        for _, row in movies_to_analyze.iterrows()
                    ]

                    results = fetch_analysis_movies(movies_with_rating, MAX_CONCURRENCY)
                    fetch_seconds = time.time() - start_time

                    all_actors_sample = []
                    genre_affinity = Counter()
                    director_affinity = Counter()
                    actor_weighted = Counter()
                    status_counts = Counter()

                    for title, year, rating, data, status in results:
                        status_counts[status] += 1
                        if not data:
                            continue

                        signed_weight = rating - NEUTRAL_RATING
                        for g in data["genres"]:
                            genre_affinity[g] += signed_weight
                        for d in data["directors"]:
                            director_affinity[d] += signed_weight

                        if signed_weight > 0:
                            all_actors_sample.extend(data["actors"])
                            for a in data["actors"]:
                                actor_weighted[a] += signed_weight

                    matched = status_counts.get("ok", 0)
                    total_movies_sampled = len(results)

                    st.caption(f"Processed {len(df)} total CSV films and analyzed taste profile in {fetch_seconds:.1f}s.")

                    if matched < total_movies_sampled:
                        failure_breakdown = ", ".join(
                            f"{count} {reason.replace('_', ' ')}"
                            for reason, count in status_counts.items()
                            if reason != "ok"
                        )
                        st.caption(f"Unmatched in sample: {failure_breakdown}.")
                        if status_counts.get("auth_error"):
                            st.error("TMDB returned an authentication error — check TMDB_API_KEY.")

                    if matched == 0:
                        st.error("Couldn't fetch data for any movies — check your TMDB API key and network connection.")
                        st.stop()

                    top_genres = [item for item, score in genre_affinity.most_common(3) if score > 0]
                    top_directors = [item for item, score in director_affinity.most_common(2) if score > 0]
                    top_actors = [item for item, _ in actor_weighted.most_common(5)]

                    avoid_genres = [g for g, score in genre_affinity.most_common() if score < -1.0][-3:]
                    avoid_directors = [d for d, score in director_affinity.most_common() if score < -1.0][-2:]

                    st.session_state.taste_profile = {
                        "genres": top_genres,
                        "directors": top_directors,
                        "actors": top_actors,
                        "avoid_genres": avoid_genres,
                        "avoid_directors": avoid_directors,
                    }

                    with st.spinner("Matching your taste profile to a character archetype..."):
                        try:
                            character_match = find_character_match_gemini(
                                st.session_state.taste_profile, FIXED_TEMPERATURE
                            )

                            local_photo = get_local_character_photo(character_match.get("character"))
                            character_match["local_photo_path"] = str(local_photo) if local_photo else None

                            st.session_state.character_match = character_match
                        except json.JSONDecodeError:
                            st.session_state.character_match = None
                        except Exception:
                            st.session_state.character_match = None

                    watchlist_titles = [w["title"] for w in st.session_state.watchlist]
                    excluded_titles = seen_movies + watchlist_titles
                    excluded_set = {normalize_title(t) for t in excluded_titles}
                    excluded_list_str = ", ".join(excluded_titles)

                    avoid_line = ""
                    if avoid_genres or avoid_directors:
                        avoid_bits = []
                        if avoid_genres:
                            avoid_bits.append(f"genres: {', '.join(avoid_genres)}")
                        if avoid_directors:
                            avoid_bits.append(f"directors: {', '.join(avoid_directors)}")
                        avoid_line = (
                            "\n                I tend to dislike movies dominated by these "
                            f"{' and '.join(avoid_bits)} — avoid recommending anything primarily in that vein.\n"
                        )

                    prompt = f"""
                    I am looking for new movie recommendations. Based on an algorithmic analysis of both my highest-rated and lowest-rated movies, here is the profile of what I enjoy most:
                    - Core Genres: {', '.join(top_genres)}
                    - Preferred Directors: {', '.join(top_directors)}
                    - Key Actors: {', '.join(top_actors)}
                    {avoid_line}
                    Recommend exactly 7 movies total, split into two groups:
                    - 5 "close_match" picks: movies that closely and safely align with my genres, directors, and actors above.
                    - 2 "adventurous_pick" picks: movies that stretch outside my usual profile — a different genre,
                      an unfamiliar director, or an unexpected style — but that someone with my taste profile would
                      still plausibly enjoy. Explain the creative leap in the reason.

                    CRITICAL CONSTRAINT: Do not recommend any movie from the following list — these have either
                    already been seen or are already on my watchlist waiting to be watched:
                    {excluded_list_str}

                    For each movie, write a "reason" of approximately 100 words explaining specifically why this
                    movie was chosen given my taste profile, and why I am likely to enjoy it. Reference the
                    genres, directors, or actors it shares with my profile where relevant, and mention anything
                    distinctive about the movie itself. For adventurous picks, explicitly note what makes it a
                    departure from my usual taste and why it's still a good bet.

                    Respond with ONLY a JSON array, no markdown code fences, no commentary, in this exact shape:
                    [
                      {{"title": "string", "year": "string (4-digit year)", "director": "string", "match_type": "close_match or adventurous_pick", "reason": "~100 word explanation"}}
                    ]
                    """

                    try:
                        client = genai.Client(api_key=GEMINI_API_KEY)
                        response = client.models.generate_content(
                            model="gemini-3.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=FIXED_TEMPERATURE),
                        )

                        raw_text = response.text.strip()
                        if raw_text.startswith("```"):
                            raw_text = raw_text.strip("`")
                            if raw_text.lower().startswith("json"):
                                raw_text = raw_text[4:].strip()

                        try:
                            recommendations = json.loads(raw_text)
                        except json.JSONDecodeError:
                            st.error("Gemini didn't return valid JSON — showing raw response instead:")
                            st.write(response.text)
                            recommendations = []

                        recommendations = [
                            r for r in recommendations
                            if normalize_title(r.get("title")) not in excluded_set
                        ]

                        if recommendations:
                            st.caption("Fetching posters, ratings, trailers, and streaming info...")
                            rec_progress = st.progress(0.0)
                            recommendations = fetch_recommendation_details(recommendations, MAX_CONCURRENCY, rec_progress)
                            st.session_state.recommendations = recommendations
                            add_history_batch(recommendations, streaming_region)
                        else:
                            st.session_state.recommendations = []
                            st.warning("No new recommendations came back after filtering out seen/watchlisted movies.")

                    except Exception as e:
                        st.error(f"Error querying Gemini API: {e}")

    # ==========================================
    # DISPLAY TASTE PROFILE + FUN FACTS + RECS
    # ==========================================
    if st.session_state.fun_facts:
        ff = st.session_state.fun_facts
        st.write("---")
        st.subheader(f"🏆 Your Ultimate Fun Facts ({ff.get('total_movies', 0)} Logged Films)")
        
        # Insert Line Chart if Data Available
        if st.session_state.monthly_watch_data is not None:
             st.markdown("### 📈 Monthly Viewing History")
             st.line_chart(st.session_state.monthly_watch_data, x="Month_Year", y="Movies Watched")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("### 🌶️ Rating & Opinion Quirks")
                if ff.get("biggest_hot_take"):
                    st.write(f"**Biggest Hater (You hated, World loved):** *{ff['biggest_hot_take']['Title']}* (You: {ff['biggest_hot_take']['User_Rating']}★ | TMDB: {ff['biggest_hot_take']['TMDB_Rating']:.1f}/10)")
                if ff.get("biggest_hidden_gem"):
                    st.write(f"**Biggest Overrate (You loved, World hated):** *{ff['biggest_hidden_gem']['Title']}* (You: {ff['biggest_hidden_gem']['User_Rating']}★ | TMDB: {ff['biggest_hidden_gem']['TMDB_Rating']:.1f}/10)")
                if ff.get("mean_rating"):
                    st.write(f"**Mean Rating:** {ff['mean_rating']:.2f}★")
                if ff.get("most_common_rating"):
                    st.write(f"**Most Generous Star:** {ff['most_common_rating']}★")

            with st.container(border=True):
                st.markdown("### ⏱️ Watch Time & Length")
                if "total_watch_days" in ff:
                    st.write(f"**Total Lifetime Watch Time:** {ff['total_watch_days']} days, {ff['total_watch_hours']} hours")
                if ff.get("avg_runtime"):
                    st.write(f"**Average Movie Length:** {ff['avg_runtime']} mins")
                if ff.get("longest_movie"):
                    st.write(f"**Longest Watch:** *{ff['longest_movie']['Title']}* ({ff['longest_movie']['Runtime']} mins)")
                if ff.get("shortest_movie"):
                    st.write(f"**Shortest Watch:** *{ff['shortest_movie']['Title']}* ({ff['shortest_movie']['Runtime']} mins)")

        with col2:
            with st.container(border=True):
                st.markdown("### 📜 Release Era & Chronology")
                if ff.get("oldest_movie"):
                    st.write(f"**Oldest Film:** *{ff['oldest_movie']['Title']}* ({ff['oldest_movie']['Year']})")
                if ff.get("newest_movie"):
                    st.write(f"**Newest Film:** *{ff['newest_movie']['Title']}* ({ff['newest_movie']['Year']})")
                if ff.get("most_watched_decade"):
                    st.write(f"**Most Watched Decade:** {ff['most_watched_decade']}s")
                if ff.get("highest_rated_decade"):
                    st.write(f"**Highest Rated Decade:** {ff['highest_rated_decade']}s")

            with st.container(border=True):
                st.markdown("### 🔍 Niche, Underground & Favorites")
                if ff.get("deepest_cut"):
                    st.write(f"**Deepest Cut:** *{ff['deepest_cut']['Title']}* (Only {ff['deepest_cut']['TMDB_Votes']} TMDB votes)")
                if ff.get("highest_community"):
                    st.write(f"**Highest Global Rating:** *{ff['highest_community']['Title']}* ({ff['highest_community']['TMDB_Rating']:.1f}/10)")
                if ff.get("most_watched_actor"):
                    st.write(f"**Most Watched Actor:** {ff['most_watched_actor'][0]} ({ff['most_watched_actor'][1]} films)")
                if ff.get("most_watched_director"):
                    st.write(f"**Most Watched Director:** {ff['most_watched_director'][0]} ({ff['most_watched_director'][1]} films)")
                
                # Insert Top Countries Break Down
                if ff.get("top_countries"):
                    country_strings = [f"{c[0]} ({c[1]})" for c in ff["top_countries"]]
                    st.write(f"**Most Watched Regions:** {', '.join(country_strings)}")

    if st.session_state.taste_profile:
        tp = st.session_state.taste_profile
        st.write("---")
        st.subheader("🧠 Your Core Taste Profile (Weighted by Rating)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Top Genres:**", ", ".join(tp["genres"]) or "—")
        with col2:
            st.write("**Top Directors:**", ", ".join(tp["directors"]) or "—")
        with col3:
            st.write("**Top Actors:**", ", ".join(tp["actors"]) or "—")

        if tp.get("avoid_genres") or tp.get("avoid_directors"):
            avoid_bits = []
            if tp.get("avoid_genres"):
                avoid_bits.append(", ".join(tp["avoid_genres"]))
            if tp.get("avoid_directors"):
                avoid_bits.append(", ".join(tp["avoid_directors"]))
            st.caption(f"🚫 Steering away from: {'; '.join(avoid_bits)}")

    if st.session_state.character_match:
        cm = st.session_state.character_match
        st.write("---")
        st.subheader("🎭 Your Character Profile")

        st.markdown(f"### You're **{cm.get('character', 'Unknown')}** from *{cm.get('movie', 'Unknown')}*")

        char_img_col, char_info_col = st.columns([2, 3])
        with char_img_col:
            if cm.get("local_photo_path"):
                st.image(cm["local_photo_path"], caption=f"{cm.get('character', 'Unknown')}")
            else:
                st.caption("No photo available")

        with char_info_col:
            if cm.get("traits"):
                st.caption(cm["traits"])
            if not cm.get("verified"):
                st.warning("⚠️ Couldn't verify this character was from the provided list — treat with a little skepticism.")
            st.write(cm.get("explanation", ""))

    if st.session_state.recommendations:
        st.write("---")
        st.subheader("🍿 Your Personalized Recommendations")

        watchlist_keys = {normalize_title(w["title"]) for w in st.session_state.watchlist}
        recs = st.session_state.recommendations

        close_matches = [r for r in recs if r.get("match_type") == "close_match"]
        adventurous = [r for r in recs if r.get("match_type") == "adventurous_pick"]
        unlabeled = [r for r in recs if r.get("match_type") not in ("close_match", "adventurous_pick")]

        if close_matches or adventurous:
            if close_matches:
                st.markdown("### 🎯 Close Matches")
                render_recommendation_grid(close_matches, "close", watchlist_keys, streaming_region, allow_refine=True)

            if adventurous:
                st.markdown("### 🎲 Adventurous Picks")
                render_recommendation_grid(adventurous, "adv", watchlist_keys, streaming_region, allow_refine=True)

            if unlabeled:
                st.markdown("### 🍿 More Recommendations")
                render_recommendation_grid(unlabeled, "unl", watchlist_keys, streaming_region, allow_refine=True)
        else:
            render_recommendation_grid(recs, "rec", watchlist_keys, streaming_region, allow_refine=True)

        st.caption(
            "Note: TMDB ratings shown are not Letterboxd's own — Letterboxd doesn't offer a public API. "
            "Streaming availability is JustWatch data via TMDB and varies in accuracy/coverage by region."
        )


def render_more_like_this_tab():
    st.write("---")
    st.subheader("🔍 More Like This")
    st.write(
        "Find movies similar to one you already know — Gemini judges similarity by tone, theme, "
        "and style, and each pick is then checked against TMDB for poster/rating/trailer/streaming/cast."
    )

    mlt_col1, mlt_col2 = st.columns([3, 1])
    with mlt_col1:
        seed_title = st.text_input("Movie title", key="mlt_title")
    with mlt_col2:
        seed_year = st.text_input("Year (optional)", key="mlt_year")

    if st.button("Find Similar Movies"):
        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            st.error("⚠️ Add your Gemini API key at the top of the script first.")
        elif not seed_title.strip():
            st.warning("Enter a movie title first.")
        else:
            gemini_error = False
            with st.spinner(f"Asking Gemini for movies similar to '{seed_title}'..."):
                try:
                    gemini_recs = find_similar_movies_gemini(
                        seed_title.strip(), seed_year.strip(), FIXED_TEMPERATURE
                    )
                except json.JSONDecodeError:
                    st.error("Gemini didn't return valid JSON — try again.")
                    gemini_recs = []
                    gemini_error = True
                except Exception as e:
                    st.error(f"Error querying Gemini API: {e}")
                    gemini_recs = []
                    gemini_error = True

            gemini_recs = [
                r for r in gemini_recs
                if normalize_title(r.get("title")) != normalize_title(seed_title)
            ]

            if not gemini_recs:
                if not gemini_error:
                    st.warning("No recommendations came back for that title.")
            else:
                st.caption("Verifying against TMDB and fetching posters, ratings, trailers, cast, and streaming info...")
                progress = st.progress(0.0)
                gemini_recs = fetch_recommendation_details(gemini_recs, MAX_CONCURRENCY, progress)
                st.session_state.more_like_this_results = {"seed": seed_title, "recs": gemini_recs}

    if st.session_state.more_like_this_results:
        data = st.session_state.more_like_this_results
        st.markdown(f"### Movies like *{data['seed']}*")
        watchlist_keys = {normalize_title(w["title"]) for w in st.session_state.watchlist}
        render_recommendation_grid(data["recs"], "mlt", watchlist_keys, streaming_region)


def render_lists_tab():
    with st.expander(f"📋 My Watchlist ({len(st.session_state.watchlist)})", expanded=False):
        if not st.session_state.watchlist:
            st.caption("Nothing on your watchlist yet. Add movies from your recommendations below.")
        else:
            for w in st.session_state.watchlist:
                wcol1, wcol2, wcol3 = st.columns([1, 3, 1])
                with wcol1:
                    if w.get("poster_path"):
                        st.image(f"{POSTER_BASE}{w['poster_path']}", width=60)
                with wcol2:
                    st.write(f"**{w['title']}** ({w.get('year', '—')})")
                    st.caption(w.get("director", ""))
                with wcol3:
                    if st.button("🗑️ Remove", key=f"remove_{normalize_title(w['title'])}"):
                        remove_from_watchlist(w["title"])
                        st.rerun()

    with st.expander(f"🕘 Recommendation History ({len(st.session_state.history)} batches)", expanded=False):
        if not st.session_state.history:
            st.caption("No past recommendation batches yet — generate some below.")
        else:
            for batch in reversed(st.session_state.history):
                st.markdown(f"**{batch['timestamp']}**")
                for rec in batch["recommendations"]:
                    hcol1, hcol2 = st.columns([1, 5])
                    with hcol1:
                        if rec.get("poster_path"):
                            st.image(f"{POSTER_BASE}{rec['poster_path']}", width=50)
                    with hcol2:
                        badge = {"close_match": "🎯", "adventurous_pick": "🎲"}.get(rec.get("match_type"), "🍿")
                        st.write(f"{badge} **{rec.get('title')}** ({rec.get('year', '—')})")
                st.divider()

            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                save_history([])
                st.rerun()


def render_random_movie_tab():
    st.subheader("🎲 Random Movie")
    st.write(
        "Roll the dice on something to watch. Filters are optional — leave them empty to pull from "
        "anything, or narrow it down by genre and by which streaming services you actually have."
    )

    if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        st.error("⚠️ Add your TMDB API key at the top of the script first.")
        return

    try:
        genres = get_genre_list(TMDB_API_KEY)
        genre_name_to_id = {g["name"]: g["id"] for g in genres}
    except httpx.HTTPError as e:
        st.error(f"Couldn't load the genre list ({e}).")
        genre_name_to_id = {}

    selected_genre_names = st.multiselect(
        "Genres (optional — leave empty for any)", sorted(genre_name_to_id.keys())
    )

    try:
        providers = get_watch_providers_list(TMDB_API_KEY, streaming_region)
    except httpx.HTTPError as e:
        st.error(f"Couldn't load streaming providers ({e}).")
        providers = []

    provider_groups = {}
    for p in providers:
        norm = normalize_provider_name(p["provider_name"])
        provider_groups.setdefault(norm, set()).add(p["provider_id"])

    selected_provider_names = st.multiselect(
        "Which streaming services do you have? (optional — start typing to search)",
        sorted(provider_groups.keys()),
        help="Start typing to filter the list. Amazon Prime variants are grouped as one option.",
    )

    if st.button("🎲 Get Random Movie"):
        genre_ids = [genre_name_to_id[n] for n in selected_genre_names]

        provider_ids = []
        for name in selected_provider_names:
            provider_ids.extend(provider_groups.get(name, []))

        with st.spinner("Rolling the dice..."):
            status, movie = asyncio.run(
                discover_random_movie_async(genre_ids, provider_ids, streaming_region, MAX_CONCURRENCY)
            )

        if status == "no_results":
            st.warning("No movies matched those filters — try loosening the genres or streaming services.")
            st.session_state.random_movie = None
        elif status != "ok":
            st.error("Couldn't fetch a random movie right now — try again.")
            st.session_state.random_movie = None
        else:
            st.session_state.random_movie = {
                "title": movie.get("title"),
                "year": movie.get("year"),
                "director": ", ".join(movie.get("directors", [])) or "Unknown",
                "genres": movie.get("genres"),
                "actors": movie.get("actors"),
                "overview": movie.get("overview"),
                "poster_path": movie.get("poster_path"),
                "vote_average": movie.get("vote_average"),
                "vote_count": movie.get("vote_count"),
                "trailer_url": movie.get("trailer_url"),
                "watch_providers": movie.get("watch_providers"),
                "verified_on_tmdb": True,
            }
            st.session_state.random_movie_provider_filter = set(selected_provider_names)

    if st.session_state.random_movie:
        st.write("---")
        watchlist_keys = {normalize_title(w["title"]) for w in st.session_state.watchlist}
        render_recommendation_card("random", st.session_state.random_movie, watchlist_keys, streaming_region)

        filter_names = st.session_state.get("random_movie_provider_filter")
        if filter_names and st.session_state.random_movie.get("watch_providers"):
            confirmed = extract_streaming_for_region(st.session_state.random_movie["watch_providers"], streaming_region)
            confirmed_flatrate = set(confirmed.get("flatrate", [])) if confirmed else set()
            if confirmed_flatrate & filter_names:
                st.success("✅ Confirmed available on one of your selected services.")
            else:
                st.info(
                    "This matched the filter at the time of discovery, but current availability may "
                    "have shifted — double check the streaming line above."
                )


# ==========================================
# TABS
# ==========================================
tab_main, tab_mlt, tab_random, tab_lists = st.tabs([
    "🔥 Trending & My Recommendations",
    "🔍 More Like This",
    "🎲 Random Movie",
    "📋 Watchlist & History",
])

with tab_main:
    render_recommendations_tab()

with tab_mlt:
    render_more_like_this_tab()

with tab_random:
    render_random_movie_tab()

with tab_lists:
    render_lists_tab()
