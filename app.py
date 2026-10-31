# Wanna Clip: find or cut viral moments from a YouTube video, with captions that pop.
import os
import re
import json
import random
import time
from collections import Counter
import shutil
import subprocess
import tempfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gradio as gr
import yt_dlp
from yt_dlp.utils import download_range_func
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from faster_whisper import WhisperModel

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1080, 1920, 30
OUT_DIR = os.path.expanduser("~/ViralClips")
os.makedirs(OUT_DIR, exist_ok=True)

_whisper = None


class _Quiet:
    def debug(self, m): pass
    def info(self, m): pass
    def warning(self, m): pass
    def error(self, m): pass


def get_whisper():
    global _whisper
    if _whisper is None:
        # "small" = good accuracy, fast on M1. First run downloads the model (~460 MB).
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# ----------------------------------------------------------------------------
# 1) FIND THE MOST VIRAL VIDEO OF THE LAST YEAR ABOUT THE TOPIC
# ----------------------------------------------------------------------------
HISTORY_FILE = os.path.join(OUT_DIR, "used_videos_v4.json")
OLD_HISTORY_FILE = os.path.join(OUT_DIR, "used_videos.json")   # written by the single-clip app
MAX_AGE_DAYS = 730  # 2 years
MAX_DURATION_ANY = 6 * 3600        # v3 listens to the whole video, so long videos are fine
MAX_DURATION_HEATMAP = 6 * 3600    # 6 h  (only the best part is downloaded)

MUSIC_WORDS = [
    "official video", "official music", "music video", "official audio", "lyric video",
    "lyrics", "(audio)", "[audio]", "official mv", " mv ", "remix", "visualizer",
    "prod.", "ft.", "feat.", "full album", "official song", "instrumental", "soundtrack",
    "theme song", "karaoke", "live performance", "official clip", "vevo", "- topic",
]
COPYRIGHT_WORDS = [
    "full movie", "full episode", "official trailer", "movie clip", "tv show",
    "full match", "full game", "match highlights", "game highlights", "full fight",
    "all rights reserved", "copyright", "\u00a9", "provided to youtube by",
    "stream rip", "netflix", "disney", "hbo",
]


RESET_DAYS = 183  # ~6 months


def load_history():
    """Used-video history. Wipes itself automatically every ~6 months."""
    today = dt.date.today()
    data = {}
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
    except Exception:
        pass
    if not data.get("_imported_old") and os.path.exists(OLD_HISTORY_FILE):   # one-time import
        try:
            with open(OLD_HISTORY_FILE) as f:
                old = json.load(f)
        except Exception:
            old = {}
        for k, v in old.items():
            if not k.startswith("_") and k not in data:
                data[k] = v

    last = None
    try:
        last = dt.date.fromisoformat(data.get("_last_reset", ""))
    except Exception:
        # old file without a date: use the file's creation date if we can
        try:
            st = os.stat(HISTORY_FILE)
            last = dt.date.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime))
        except Exception:
            last = today

    if (today - last).days >= RESET_DAYS:
        data, last = {}, today          # 6 months passed -> clear everything

    data["_imported_old"] = True
    data["_last_reset"] = last.isoformat()
    # migrate old entries (video id -> title) to the richer format
    for k, v in list(data.items()):
        if not k.startswith("_") and not isinstance(v, dict):
            data[k] = {"title": str(v), "url": f"https://www.youtube.com/watch?v={k}",
                       "channel": "", "moments": [], "last_used": None, "legacy": True}
    save_history(data)
    return data


def fmt_time(sec):
    sec = int(sec)
    h, m, x = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{x:02d}" if h else f"{m}:{x:02d}"


def new_entry(info):
    return {"title": info.get("title", ""), "url": info.get("webpage_url") or
            f"https://www.youtube.com/watch?v={info['id']}",
            "channel": info.get("channel") or info.get("uploader") or "",
            "moments": [], "last_used": None, "legacy": False}


def get_entry(info):
    h = load_history()
    e = h.get(info["id"])
    if not isinstance(e, dict):
        e = new_entry(info)
        h[info["id"]] = e
        save_history(h)
    return e


def record_use(info, used_moments, start):
    """Remember that the moment starting at `start` was used (shows in 'Last used')."""
    h = load_history()
    e = h.get(info["id"]) or new_entry(info)
    e["moments"] = list(used_moments) + [int(start)]
    e["legacy"] = False
    e["last_used"] = dt.datetime.now().isoformat(timespec="seconds")
    h[info["id"]] = e
    save_history(h)


def recent_videos(n=3):
    h = load_history()
    items = [(v["last_used"], k, v) for k, v in h.items()
             if not k.startswith("_") and isinstance(v, dict) and v.get("last_used")]
    items.sort(key=lambda t: t[0], reverse=True)
    return [{"id": k, **v} for _, k, v in items[:n]]


def save_history(h):
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=1)


def is_music_or_copyrighted(info):
    title = (info.get("title") or "").lower()
    chan = (info.get("channel") or info.get("uploader") or "").lower()
    desc = (info.get("description") or "")[:1500].lower()
    cats = [c.lower() for c in (info.get("categories") or [])]
    tags = " ".join(info.get("tags") or []).lower()

    # --- music ---
    if "music" in cats:
        return "music"
    if info.get("track") or info.get("artist") or info.get("album"):
        return "music"
    if any(w in title for w in MUSIC_WORDS) or "vevo" in chan or chan.endswith("- topic"):
        return "music"
    if "provided to youtube by" in desc or "auto-generated by youtube" in desc:
        return "music"
    if any(w in tags for w in ("official music video", "lyrics", "lyric video")):
        return "music"

    # --- likely copyrighted / claimed content ---
    if any(w in title for w in COPYRIGHT_WORDS) or any(w in desc for w in COPYRIGHT_WORDS):
        return "copyrighted"
    if info.get("live_status") in ("is_live", "was_live") and "stream" in title:
        return "livestream"
    return None


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "without",
    "from", "by", "about", "is", "are", "was", "were", "be", "my", "your", "his", "her",
    "their", "our", "me", "him", "them", "this", "that", "it", "vs", "versus", "video",
    "videos", "best", "funny", "new", "when", "how", "what", "why", "who",
}


def keywords(query):
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) >= 2]
    kws = [w for w in words if w not in STOPWORDS]
    return kws or words


def _stem(k):
    for suf in ("ing", "ion", "ed", "es", "s"):
        if k.endswith(suf) and len(k) - len(suf) >= 4:
            return k[: -len(suf)]
    return k


def relevance(query, *texts):
    """0..1 : how much of the search sentence is covered by the given text."""
    kws = keywords(query)
    if not kws:
        return 0.0
    text = " ".join(t or "" for t in texts)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    joined = "".join(tokens)

    def hit(k):
        st = _stem(k)
        for t in tokens:
            if t == k or t.rstrip("s") == k.rstrip("s"):
                return True
            if len(st) >= 4 and t.startswith(st):        # react -> reacts / reaction
                return True
        return len(k) >= 6 and k in joined               # "somename" == "Some Name"

    score = sum(1 for k in kws if hit(k)) / len(kws)
    if norm(query) in norm(text):          # whole sentence appears -> perfect
        score = 1.0
    return score


# Titles/tags that signal "someone reacting to / reviewing / commenting on another video"
REACTION_RE = re.compile(
    r"\b(react\w*|reaction\w*|respond\w*|response to|reviewing|my thoughts|"
    r"watching|watch party|watch along|watchalong)\b", re.I)


def subject_terms(query, own_channel=""):
    """Name(s) that identify the subject's OWN channel (e.g. 'somename')."""
    manual = [norm(x) for x in re.split(r"[,;]", own_channel or "") if norm(x)]
    if manual:
        return manual
    kws = [k for k in keywords(query) if not k.startswith("react")]
    # the longest keyword is almost always the person/brand name
    return [max(kws, key=len)] if kws else []


def is_own_channel(channel, terms):
    c = norm(channel)
    return any(t and t in c for t in terms)


def is_third_party_reaction(title, tags, channel, terms):
    """True if it's somebody ELSE's reaction/review video (their own original content)."""
    text = f"{title or ''} {' '.join(tags or [])}"
    return bool(REACTION_RE.search(text)) and not is_own_channel(channel, terms)


def is_youtube_url(url):
    return bool(re.search(r"(youtube\.com|youtu\.be)/", url or "", re.I))


def get_video_info(url):
    """Open a specific video. Falls back to your browser login (age-restricted videos etc.)."""
    last = None
    for extra in ({}, {"cookiesfrombrowser": ("brave",)}, {"cookiesfrombrowser": ("chrome",)},
                  {"cookiesfrombrowser": ("safari",)}, {"cookiesfrombrowser": ("firefox",)}):
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "logger": _Quiet(),
                                   "skip_download": True, "noplaylist": True, **extra}) as y:
                return y.extract_info(url, download=False)
        except Exception as e:
            last = e
    raise RuntimeError(f"Couldn't open that video: {last}")


def use_link(url):
    """Direct link: the user chose this video, so no filters are applied."""
    if not is_youtube_url(url):
        raise RuntimeError("That doesn't look like a YouTube link.")
    info = get_video_info(url)
    if not info.get("duration"):
        raise RuntimeError("That video is live right now (or has no length). Try again after it ends.")
    get_entry(info)          # creates the history entry if this is a new video
    return info


def find_video(query, cc_only=False, min_views=100_000, own_channel=""):
    """Pick ONE random, never-used, on-topic, recent, popular, non-music video.
    Rejects other people's reaction videos and anything under `min_views`.
    If the search text is the exact title of a video, that video is used directly."""
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "logger": _Quiet(),
                           "extract_flat": True, "skip_download": True}) as y:
        res = y.extract_info(f"ytsearch100:{query}", download=False)

    raw = [e for e in (res.get("entries") or []) if e and e.get("id")]

    # exact title typed -> the person knows which video they want
    nq = norm(query)
    if len(nq) >= 12:
        for e in raw:
            if norm(e.get("title")) == nq:
                return use_link(f"https://www.youtube.com/watch?v={e['id']}")

    history = load_history()
    terms = subject_terms(query, own_channel)
    min_views = int(min_views or 0)
    rej = Counter()   # why videos were skipped (shown if nothing is found)

    def chan(e):
        return e.get("channel") or e.get("uploader") or ""

    # 1) hard filters FIRST, so the best-match group is chosen from usable videos only
    entries = []
    for e in raw:
        if e["id"] in history:
            rej["already used before"] += 1
        elif e.get("duration") and not 120 <= e["duration"] <= MAX_DURATION_HEATMAP:
            rej["too short/too long"] += 1
        elif e.get("view_count") is not None and e["view_count"] < min_views:
            rej[f"under {min_views:,} views"] += 1
        elif is_third_party_reaction(e.get("title"), None, chan(e), terms):
            rej["someone else's reaction video"] += 1
        elif is_music_or_copyrighted({"title": e.get("title"), "channel": chan(e)}):
            rej["music / likely copyrighted"] += 1
        else:
            entries.append(e)

    # 2) relevance to the search sentence: keep only the best-matching group
    def rel(e):
        return relevance(query, e.get("title"), chan(e))

    scored = [(rel(e), e) for e in entries]
    best = max((r for r, _ in scored), default=0.0)
    if best > 0:
        thr = max(min(best, 0.6), best - 0.2)
        pool = [e for r, e in scored if r >= thr]
    else:
        pool = [e for _, e in scored][:20]
    rej["not matching your words well enough"] += len(entries) - len(pool)

    # 3) popular first (top 40 by views), then shuffle -> random every time
    pool.sort(key=lambda e: e.get("view_count") or 0, reverse=True)
    pool = pool[:40]
    random.shuffle(pool)

    cutoff = (dt.date.today() - dt.timedelta(days=MAX_AGE_DAYS)).strftime("%Y%m%d")
    for e in pool:
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "logger": _Quiet(),
                                   "skip_download": True, "noplaylist": True}) as y:
                info = y.extract_info(f"https://www.youtube.com/watch?v={e['id']}", download=False)
        except Exception:
            rej["couldn't be opened (age-restricted/unavailable)"] += 1
            continue
        dur = info.get("duration") or 0
        if (info.get("upload_date") or "0") < cutoff:
            rej["older than 2 years"] += 1
            continue
        if not 120 <= dur <= (MAX_DURATION_HEATMAP if info.get("heatmap") else MAX_DURATION_ANY):
            rej["too short/too long"] += 1
            continue
        if (info.get("view_count") or 0) < min_views:
            rej[f"under {min_views:,} views"] += 1
            continue
        ch = info.get("channel") or info.get("uploader") or ""
        if is_third_party_reaction(info.get("title"), info.get("tags"), ch, terms):
            rej["someone else's reaction video"] += 1
            continue
        if is_music_or_copyrighted(info):
            rej["music / likely copyrighted"] += 1
            continue
        if cc_only and "creative commons" not in (info.get("license") or "").lower():
            rej["not Creative Commons"] += 1
            continue
        get_entry(info)
        return info

    why = ", ".join(f"{n} {r}" for r, n in rej.items() if n) or "no search results"
    raise RuntimeError(
        f"No matching video found (min {min_views:,} views). Skipped: {why}. "
        "Tip: lower 'Minimum views', use broader words, or paste the video's link instead.")


# ----------------------------------------------------------------------------
# 2) DOWNLOAD
# ----------------------------------------------------------------------------
AUDIO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".opus", ".mp3", ".aac", ".ogg")
VIDEO_FMT = "bv*[height<=720]+ba/b[height<=720]/b"
VIDEO_FMT_HD = "bv*[height<=1080]+ba/b[height<=1080]/b"      # regular (horizontal) videos
AUDIO_FMT = "wa/worstaudio/ba/b"          # smallest audio: enough to find the moments


def _find_downloaded(workdir, prefix="source"):
    files = [f for f in os.listdir(workdir)
             if f.startswith(prefix) and f.lower().endswith(AUDIO_EXTS)]
    files.sort(key=lambda f: os.path.getsize(os.path.join(workdir, f)), reverse=True)
    return os.path.join(workdir, files[0]) if files else None


def _download_try(info, workdir, section, name="source", fmt=VIDEO_FMT):
    base = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(workdir, name + ".%(ext)s"),
        "ffmpeg_location": FFMPEG,
        "retries": 10, "fragment_retries": 10,
        "http_chunk_size": 10 * 1024 * 1024,   # smaller chunks avoid many 403s
    }
    if section:                                  # download ONLY this part of the video
        base["download_ranges"] = download_range_func(None, [section])
        base["force_keyframes_at_cuts"] = True
    attempts = [
        {},
        {"extractor_args": {"youtube": {"player_client": ["tv", "web_safari"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android", "ios"]}}},
        {"cookiesfrombrowser": ("brave",)},
        {"cookiesfrombrowser": ("chrome",)},
        {"cookiesfrombrowser": ("safari",)},
        {"cookiesfrombrowser": ("firefox",)},
    ]
    last_err = None
    for extra in attempts:
        for f in os.listdir(workdir):           # clean partial files of THIS download
            if f.startswith(name):
                try:
                    os.remove(os.path.join(workdir, f))
                except OSError:
                    pass
        try:
            with yt_dlp.YoutubeDL({**base, **extra}) as y:
                y.download([info["webpage_url"]])
            path = _find_downloaded(workdir, name)
            if path:
                return path
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All download methods failed (YouTube blocked it). Last error: {last_err}")


def download(info, workdir, section=None, fmt=VIDEO_FMT):
    """Returns (file, offset). With a section, the file starts at `offset` seconds of the video."""
    if section:
        try:
            return _download_try(info, workdir, section, fmt=fmt), section[0]
        except Exception:
            pass                                 # fall back to the full video below
    return _download_try(info, workdir, None, fmt=fmt), 0


# ----------------------------------------------------------------------------
# 4) TRANSCRIBE THE CLIP (word-level timestamps)
# ----------------------------------------------------------------------------
def transcribe(src, start, dur, workdir):
    # Decode audio ourselves with the bundled ffmpeg -> numpy array (no PyAV needed)
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        stdout=subprocess.PIPE, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    segments, _ = get_whisper().transcribe(audio, word_timestamps=True, vad_filter=True)
    words = []
    for s_ in segments:
        for w in (s_.words or []):
            t = w.word.strip()
            if t:
                words.append((t, w.start, w.end))
    return words


def make_groups(words, size=3):
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= size or w[1] - cur[-1][2] > 0.6):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,]$", w[0]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        nxt = groups[i + 1][0][1] if i + 1 < len(groups) else g[-1][2] + 0.4
        out.append({"start": g[0][1], "end": min(nxt, g[-1][2] + 0.4), "words": g})
    return out


# ----------------------------------------------------------------------------
# 5) VIRAL CAPTION STYLE  (big bold uppercase, black stroke, active word yellow)
# ----------------------------------------------------------------------------
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
STRIP_H, STRIP_Y = 280, 1330
FORMATS = {
    # w, h = output size · base = caption font size · strip_y/strip_h = caption position
    # side = free space left and right · words = words per caption · max_len = longest allowed (seconds)
    "short":   dict(w=1080, h=1920, base=120, strip_h=280, strip_y=1330, side=100, words=3, max_len=180),
    "regular": dict(w=1920, h=1080, base=64, strip_h=170, strip_y=800, side=320, words=4, max_len=600),
}
YELLOW, WHITE = (255, 222, 0, 255), (255, 255, 255, 255)


def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size)      # Pillow 10.1+ can scale the built-in font
    except TypeError:
        return ImageFont.load_default()


_strip_cache = {}


def caption_strip(texts, active, pop=1.0, style=None):
    """Caption image for one group of words. `pop` > 1 enlarges the active word (pop animation)."""
    st = style or FORMATS["short"]
    cw = st["w"]
    key = (tuple(texts), active, round(pop, 2), cw)
    if key in _strip_cache:
        return _strip_cache[key]
    texts = [t.upper() for t in texts]
    size = st["base"]
    while True:
        fonts = [load_font(int(size * (pop if i == active else 1.0))) for i in range(len(texts))]
        space = fonts[0].getlength(" ")
        widths = [f.getlength(t) for f, t in zip(fonts, texts)]
        total = sum(widths) + space * (len(texts) - 1)
        if total <= cw - st["side"] or size <= st["base"] * 0.4:
            break
        size -= max(2, st["base"] // 20)
    img = Image.new("RGBA", (cw, st["strip_h"]), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = (cw - total) / 2
    for i, t in enumerate(texts):
        fsz = int(size * (pop if i == active else 1.0))
        d.text((x, st["strip_h"] // 2), t, font=fonts[i], anchor="lm",
               fill=YELLOW if i == active else WHITE,
               stroke_width=max(4, fsz // 11), stroke_fill=(0, 0, 0, 255))
        x += widths[i] + space
    arr = np.array(img)
    res = (arr[..., :3].astype(np.uint16), arr[..., 3:4].astype(np.uint16))
    _strip_cache[key] = res
    return res


def blend(frame, rgb, alpha, y0):
    h = rgb.shape[0]
    y1 = min(frame.shape[0], y0 + h)
    reg = frame[y0:y1].astype(np.uint16)
    frame[y0:y1] = ((reg * (255 - alpha[:y1 - y0]) + rgb[:y1 - y0] * alpha[:y1 - y0]) // 255).astype(np.uint8)


FX_KEYS = ["glitch", "pop"]
FX_DEFAULTS = dict(glitch=True, pop=True)


# ============================================================================
# V4: listen to the WHOLE video -> find sections -> key moments (football: goals found with
#     referee whistles + words + crowd noise) -> cut -> join in video order -> pop captions
#     + glitch transitions.
# ============================================================================
ANALYSIS_SR = 16000
ANALYSIS_HOP = 0.25            # seconds per loudness sample
CHUNK_SECONDS = 600            # audio is analysed 10 minutes at a time (low memory)
MAX_ANALYZE_SECONDS = 4 * 3600
ANALYSIS_MODEL = "tiny"        # fast transcript used only to FIND moments ("base" = slower, more accurate)
MIN_FILL, TARGET_FILL = 0.88, 0.97   # shorter than 88% of the length asked -> pad moments up to ~97% (never above 100%)
FB_DENSITY = 1.2               # football words per minute that make a section "football"
WH_PER_MIN = 0.4               # ...or this many whistles per minute
GOAL_MIN_WIN, GOAL_MAX_WIN = 8.0, 18.0   # seconds kept around each goal (shrinks to fit more goals)
MOMENT_PRE, MOMENT_POST = 8.0, 8.0       # seconds kept around a normal moment
MATCH_GAP = 110.0              # no whistle for this long (seconds) = a different match
SR = 44100
SPF = SR // FPS                # audio samples per video frame (1470 at 30 fps)

_whisper_tiny = None


def get_whisper_tiny():
    global _whisper_tiny
    if _whisper_tiny is None:
        _whisper_tiny = WhisperModel(ANALYSIS_MODEL, device="cpu", compute_type="int8")
    return _whisper_tiny


# ---------------- audio helpers ----------------
def decode_mono16k(path, start, dur):
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", path,
         "-vn", "-ac", "1", "-ar", str(ANALYSIS_SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, dtype=np.float32)


def decode_stereo(src, start, dur):
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src,
         "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    a = a[: len(a) // 2 * 2].reshape(-1, 2).copy()
    need = (int(dur * FPS) + 2) * SPF
    if len(a) < need:
        a = np.concatenate([a, np.zeros((need - len(a), 2), np.float32)])
    return a[:need]


def rolling_median(x, win):
    pad = win // 2
    xp = np.pad(x, (pad, win - pad - 1), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, win), axis=1)


def normalize_audio(a, target_db=-18.0):
    mono = a.mean(axis=1)
    gate = np.abs(mono) > 0.01
    if gate.sum() < len(mono) * 0.01:
        return a
    rms = float(np.sqrt(np.mean(mono[gate] ** 2)))
    gain = float(np.clip(10 ** ((target_db - 20 * np.log10(rms + 1e-9)) / 20), 0.35, 5.0))
    return a * gain


def soft_limit(a):
    return (0.95 * np.tanh(a / 0.95)).astype(np.float32)


def write_wav(path, a):
    import wave
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def add_sfx(track, sound, at_frame, gain):
    s = at_frame * SPF
    if s < 0 or s >= len(track):
        return
    seg = sound[: len(track) - s] * gain
    track[s: s + len(seg)] += seg


# ---------------- referee whistle detector ----------------
WH_N, WH_HOP = 1024, 512            # 64 ms windows, 32 ms step at 16 kHz
WH_LO_HZ, WH_HI_HZ = 2000, 4500     # a referee whistle lives in this band


def detect_whistles(pcm, offset=0.0, sr=ANALYSIS_SR):
    """Referee whistles = one very narrow, steady, loud tone between 2 and 4.5 kHz.
    Voices, crowds and music are not like that (they are spread over many frequencies).
    Returns [{'t': start_second, 'end': end_second, 'strength': 0..1}]."""
    pcm = np.ascontiguousarray(pcm, dtype=np.float32)
    n = len(pcm)
    if n < WH_N * 8:
        return []
    nf = (n - WH_N) // WH_HOP + 1
    frames = np.lib.stride_tricks.as_strided(
        pcm, shape=(nf, WH_N), strides=(pcm.strides[0] * WH_HOP, pcm.strides[0]))
    win = np.hanning(WH_N).astype(np.float32)
    lo, hi = int(WH_LO_HZ * WH_N / sr), int(WH_HI_HZ * WH_N / sr)
    flag = np.zeros(nf, bool)
    strength = np.zeros(nf, np.float32)
    for b0 in range(0, nf, 3000):
        blk = frames[b0:b0 + 3000] * win
        p = (np.abs(np.fft.rfft(blk, axis=1)) / (WH_N / 4)) ** 2
        total = p[:, 4:].sum(axis=1) + 1e-12
        band = p[:, lo:hi]
        bsum = band.sum(axis=1) + 1e-12
        kmax = band.argmax(axis=1)
        rows = np.arange(band.shape[0])
        pk7 = np.zeros(band.shape[0])
        for d in range(-3, 4):                            # energy of the peak +-3 bins
            pk7 += band[rows, np.clip(kmax + d, 0, band.shape[1] - 1)]
        ratio = bsum / total                              # how much of the sound is in the whistle band
        dominance = pk7 / bsum                            # how much of that is ONE narrow tone
        peak_amp = np.sqrt(pk7)
        ok = (ratio > 0.30) & (dominance > 0.55) & (peak_amp > 0.01)
        flag[b0:b0 + 3000] = ok
        strength[b0:b0 + 3000] = np.where(ok, np.minimum(1.0, ratio * dominance * 1.5), 0)
    events, i = [], 0
    while i < nf:
        if not flag[i]:
            i += 1
            continue
        j = i
        gap = 0
        last = i
        while j < nf and gap <= 2:                        # allow tiny dropouts (the pea trill)
            if flag[j]:
                last = j
                gap = 0
            else:
                gap += 1
            j += 1
        dur = (last - i + 1) * WH_HOP / sr
        if 0.2 <= dur <= 4.0:
            t0 = offset + (i * WH_HOP + WH_N / 2) / sr
            events.append({"t": float(t0), "end": float(t0 + dur),
                           "strength": float(strength[i:last + 1].mean())})
        i = last + 1
    return events


def merge_blasts(whistles, gap=1.5):
    """Double / triple blasts count as ONE whistle moment."""
    out = []
    for w in sorted(whistles, key=lambda w: w["t"]):
        if out and w["t"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], w["end"])
            out[-1]["strength"] = max(out[-1]["strength"], w["strength"])
        else:
            out.append(dict(w))
    return out


# ---------------- listen to the whole video ----------------
def analyze_video(path, total):
    """Generator. Yields progress text, returns {db, hop, segments, whistles, analyzed}.
    Loudness curve + fast transcript + referee whistles for the WHOLE audio."""
    dur = int(min(total, MAX_ANALYZE_SECONDS))
    model = get_whisper_tiny()
    db_parts, segments, whistles, lang = [], [], [], None
    t0 = time.time()
    for c0 in range(0, dur, CHUNK_SECONDS):
        clen = min(CHUNK_SECONDS, dur - c0)
        frac = c0 / dur
        eta = ""
        if frac > 0.05:
            left = (time.time() - t0) / frac * (1 - frac)
            eta = f" · about {int(left // 60) + 1} min left"
        yield f"🎧 Listening to the whole video… {int(frac * 100)}%{eta}"
        pcm = decode_mono16k(path, c0, clen)
        want = int(round(clen / ANALYSIS_HOP))
        n = int(ANALYSIS_HOP * ANALYSIS_SR)
        m = len(pcm) // n
        if m == 0:
            db_parts.append(np.full(want, -90.0))
            continue
        rms = np.sqrt((pcm[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
        part = 20 * np.log10(rms + 1e-9)
        if len(part) < want:
            part = np.concatenate([part, np.full(want - len(part), part[-1])])
        db_parts.append(part[:want])
        whistles.extend(detect_whistles(pcm, offset=c0))
        segs, info = model.transcribe(pcm, language=lang, vad_filter=True, beam_size=1,
                                      condition_on_previous_text=False)
        if lang is None:
            lang = info.language
        for s in segs:
            segments.append((c0 + s.start, c0 + s.end, s.text.strip()))
    db = np.concatenate(db_parts) if db_parts else np.zeros(0)
    return {"db": db, "hop": ANALYSIS_HOP, "segments": segments,
            "whistles": merge_blasts(whistles), "analyzed": dur}


# ---------------- loud moments ----------------
def find_energy_peaks(db, hop, thr=6.0, baseline_s=20.0, sep_s=8.0, max_n=800):
    """Sudden loud moments compared with the surrounding ~20 seconds.
    'sus' = how many seconds the loudness stays up (a crowd roar lasts, a click does not)."""
    n = len(db)
    if n < 40:
        return []
    sm = np.convolve(db, np.ones(3) / 3, mode="same")
    base = rolling_median(sm, max(5, int(baseline_s / hop)))
    excess = sm - base
    k = int(1.0 / hop)
    cand = []
    for i in np.where((excess >= thr) & (sm > -48))[0]:
        lo, hi = max(0, i - k), min(n, i + k + 1)
        if sm[i] >= sm[lo:hi].max():
            cand.append((float(excess[i]), int(i)))
    cand.sort(reverse=True)
    chosen = []
    for ex, i in cand:
        if all(abs(i - j) * hop >= sep_s for _, j in chosen):
            chosen.append((ex, i))
        if len(chosen) >= max_n:
            break
    peaks = []
    for ex, i in chosen:
        j, lim = i, max(0, i - int(3.0 / hop))
        while j > lim and excess[j] > 0.5 * ex:
            j -= 1
        sus = float((excess[max(0, i - int(2 / hop)): i + int(6 / hop)] > max(3.0, 0.5 * ex)).sum() * hop)
        peaks.append({"peak": i * hop, "on": j * hop, "excess": ex, "sus": sus})
    peaks.sort(key=lambda p: p["peak"])
    return peaks


# ---------------- words ----------------
STOP = set("""the and for that this with you your have are was were but not they them their then than
there what when where which who whom why how can will just like really very get got going gonna want know
think yeah yes okay all out about from has had his her him she its our into over been being some more
most much many any does did doing done also would could should only even still such own same too because
while after before again here those these other another bro man dude guys guy look right now let come make
made say said see saw one two""".split())

FOOTBALL_WORDS = set("""goal goals penalty penalties striker midfield midfielder defender defenders goalkeeper
keeper pass passes shoot shot shots header freekick corner offside referee halftime fulltime kickoff kick match
fifa fc ultimate tackle dribble cross assist whistle pitch possession formation score scored scores foul fouled
yellow card""".split())

GOAL_RE = re.compile(r"\b(go+a?l+s?|golazo|scored|scores|si+u+|back of the net|top bins?|"
                     r"what a finish|what a strike|what a shot)\b")
FOUL_RE = re.compile(r"\b(fou+l\w*|fault|referee|ref|yellow card|red card|booked|free[- ]?kick|offside|"
                     r"hand ?ball|dive|diving|var)\b")
# excited OR frustrated reactions (a goal makes people shout either way)
EXCITE_RE = re.compile(r"\b(oh+|wow|omg|no+|no way|crazy|insane|unbelievable|let'?s go+|yes+|what the|"
                       r"come on|are you serious|seriously|screaming|scream|ugh+)\b")
END_RE = re.compile(r"\b(full[- ]?time|final whistle|game over|match over|rematch|next (match|game)|"
                    r"play (again|another)|another (match|game))\b")


def tokenize(text):
    return [w for w in re.findall(r"[a-z']{3,}", text.lower()) if w not in STOP]


def text_between(segments, a, b):
    return " ".join(t for s, e, t in segments if e >= a and s <= b).lower()


def football_density(segments, a, b):
    mins = max(1.0, (b - a) / 60.0)
    n = sum(1 for s, e, t in segments if a <= s < b
            for w in re.findall(r"[a-z]+", t.lower()) if w in FOOTBALL_WORDS)
    return n / mins


# ---------------- football: goals vs fouls, using whistles ----------------
def classify_football_peaks(peaks, segments, whistles):
    """Label every loud moment: goal / foul / other.
      foul : a whistle right at the moment (play stopped) or the word foul/ref/card... nearby
      goal : 'goal'/'scored'/'siuu' said, OR a long crowd roar followed a few seconds later by the
             restart whistle with excited/frustrated speech (no whistle at the moment itself)."""
    wt = np.array(sorted(w["t"] for w in whistles))
    has_wh = len(wt) > 0
    for p in peaks:
        t = p["peak"]
        txt = text_between(segments, t - 6, t + 5)
        p["goal_w"] = len(GOAL_RE.findall(txt))
        p["foul_w"] = len(FOUL_RE.findall(text_between(segments, t - 10, t + 10)))
        p["excite"] = len(EXCITE_RE.findall(txt)) + txt.count("!")
        p["wh_now"] = bool(has_wh and np.any((wt >= t - 3.0) & (wt <= t + 3.0)))
        p["wh_after"] = bool(has_wh and np.any((wt > t + 4.0) & (wt <= t + 45.0)))
        if p["goal_w"] > 0 and p["goal_w"] >= p["foul_w"]:
            kind = "goal"
        elif p["foul_w"] > 0 or p["wh_now"]:
            kind = "foul"
        elif p["wh_after"] and p["sus"] >= 3.0 and (p["excite"] > 0 or p["excess"] >= 6.0):
            kind = "goal"
        elif (not has_wh) and p["excess"] >= 8.0:       # no whistle information at all
            kind = "goal"
        else:
            kind = "other"
        p["kind"] = kind
        p["gscore"] = (1.0 + 1.5 * (p["goal_w"] > 0) + 0.7 * p["wh_after"]
                       + min(p["excess"], 16) / 16 + 0.3 * min(p["excite"], 3))


def split_matches(goals, whistles, segments, a, b):
    """Group goals into matches. A long stretch without any whistle (or 'full time' / 'next match')
    means a new match started."""
    goals = sorted(goals, key=lambda g: g["peak"])
    ends = [(s + e) / 2 for s, e, t in segments if a <= s <= b and END_RE.search(t.lower())]
    wt = sorted(w["t"] for w in whistles if a <= w["t"] <= b)
    if len(wt) >= 4:
        spans = [[wt[0], wt[0]]]
        for t in wt[1:]:
            if t - spans[-1][1] > MATCH_GAP or any(spans[-1][1] < x < t for x in ends):
                spans.append([t, t])
            else:
                spans[-1][1] = t
        spans = [s for s in spans if s[1] - s[0] >= 20] or spans
        if len(spans) >= 2:
            groups = [[] for _ in spans]
            for g in goals:
                d = [0 if s[0] - 30 <= g["peak"] <= s[1] + 60 else min(abs(g["peak"] - s[0]), abs(g["peak"] - s[1]))
                     for s in spans]
                groups[int(np.argmin(d))].append(g)
            return [grp for grp in groups if grp]
    clusters, cur = [], []                       # fallback: big gaps between goals
    for g in goals:
        if cur and (g["peak"] - cur[-1]["peak"] > 720 or any(cur[-1]["peak"] < x < g["peak"] for x in ends)):
            clusters.append(cur)
            cur = []
        cur.append(g)
    if cur:
        clusters.append(cur)
    return clusters


def goal_windows(goals, max_len):
    """One short window per goal, in the order the goals happened (never sorted by reaction size).
    Many goals -> shorter windows (down to 8 s each); too many for the limit -> keep the clearest."""
    keep = goals
    if len(keep) * GOAL_MIN_WIN > max_len:
        k = max(1, int(max_len // GOAL_MIN_WIN))
        keep = sorted(goals, key=lambda g: -g["gscore"])[:k]
    keep = sorted(keep, key=lambda g: g["peak"])
    cap = GOAL_MAX_WIN if max_len <= 120 else min(75.0, GOAL_MAX_WIN + (max_len - 120) * 0.5)   # longer videos: longer goal clips
    L = min(cap, max(GOAL_MIN_WIN, max_len / len(keep)))
    return [(g["peak"] - 0.65 * L, g["peak"] + 0.35 * L, g["gscore"]) for g in keep]


# ---------------- topic sections (does the video change subject?) ----------------
def _cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb + 1e-9)


def _merge_short(sections, min_len):
    secs = [list(s) for s in sections]
    while len(secs) > 1:
        lens = [b - a for a, b in secs]
        i = int(np.argmin(lens))
        if lens[i] >= min_len:
            break
        j = i - 1 if i > 0 else i + 1
        a = min(secs[i][0], secs[j][0])
        b = max(secs[i][1], secs[j][1])
        secs[min(i, j)] = [a, b]
        del secs[max(i, j)]
    return [tuple(s) for s in secs]


def segment_topics(segments, duration, chapters=None, block=60, min_section=300):
    """Split the video into sections by subject: YouTube chapters if the creator made them,
    otherwise by how much the spoken vocabulary changes."""
    if chapters and len(chapters) >= 2:
        secs = [(float(c["start_time"]), float(c["end_time"])) for c in chapters]
        return _merge_short(secs, 180)
    if duration < 720 or len(segments) < 20:
        return [(0.0, float(duration))]
    nb = int(np.ceil(duration / block))
    blocks = [Counter() for _ in range(nb)]
    for s, e, t in segments:
        blocks[min(nb - 1, int(s // block))].update(tokenize(t))
    sims = {}
    for i in range(1, nb):
        left, right = Counter(), Counter()
        for b in blocks[max(0, i - 5): i]:
            left.update(b)
        for b in blocks[i: i + 5]:
            right.update(b)
        sims[i] = _cosine(left, right) if sum(left.values()) >= 10 and sum(right.values()) >= 10 else None
    vals = [v for v in sims.values() if v is not None]
    if len(vals) < 6:
        return [(0.0, float(duration))]
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if sd < 0.02:
        return [(0.0, float(duration))]
    thr = mu - 0.6 * sd
    cands = []
    for i, v in sims.items():
        if v is None or v >= thr:
            continue
        near = [sims[j] for j in range(max(1, i - 3), min(nb, i + 4)) if sims.get(j) is not None]
        if v <= min(near):
            cands.append((v, i))
    cands.sort()
    gap = max(5, int(min_section / block))
    chosen = []
    for v, i in cands:
        if all(abs(i - j) >= gap for j in chosen):
            chosen.append(i)
    bounds = [0] + sorted(chosen) + [nb]
    secs = [(float(bounds[k] * block), float(min(duration, bounds[k + 1] * block)))
            for k in range(len(bounds) - 1)]
    return _merge_short(secs, min_section)


# ---------------- windows: merge (always in video order), fit, pad ----------------
def merge_windows(ws, gap=1.0):
    out = []
    for a, b in sorted([list(w) for w in ws]):          # sorted by TIME = original video order
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total_len(ws):
    return sum(b - a for a, b in ws)


GAP_KEEP = 6.0     # seconds kept between two moments while padding, so they stay separate cuts


def pad_windows(ws, lo, hi, target):
    """Grow every moment before and after until the total reaches `target` seconds.
    First the moments stay separate cuts (a small gap is kept); only if that cannot reach the
    target do neighbouring moments grow into each other and merge."""
    ws = [list(w) for w in ws]
    for _ in range(3000):
        tot = total_len(ws)
        if tot >= target - 0.5:
            return ws
        grow = []
        for i, (a, b) in enumerate(ws):
            left = a - (ws[i - 1][1] + GAP_KEEP if i > 0 else lo)
            right = (ws[i + 1][0] - GAP_KEEP if i < len(ws) - 1 else hi) - b
            grow.append((min(2.0, max(0.0, left)), min(2.0, max(0.0, right))))
        possible = sum(l + r for l, r in grow)
        if possible < 0.01:
            break
        scale = min(1.0, (target - tot) / possible)
        for i, (l, r) in enumerate(grow):
            ws[i][0] -= l * scale
            ws[i][1] += r * scale
    for _ in range(3000):                             # still short: let neighbours touch and merge
        tot = total_len(ws)
        if tot >= target - 0.5:
            break
        step = min(2.0, (target - tot) / (2 * len(ws)))
        if step < 0.05:
            break
        new = [[max(lo, a - step), min(hi, b + step)] for a, b in ws]
        if all(abs(n[0] - w[0]) < 1e-6 and abs(n[1] - w[1]) < 1e-6 for n, w in zip(new, ws)):
            break
        ws = merge_windows(new, gap=0.0)
    return ws


def fit_windows(items, lo, hi, max_len):
    """items: [(start, end, score)] -> windows in video order, never longer than max_len in total;
    if they come out too short they are padded before/after (still never above max_len)."""
    items = [(max(lo, a), min(hi, b), s) for a, b, s in items if min(hi, b) - max(lo, a) >= 3]
    if not items:
        return []
    gap = 1.0 if max_len <= 120 else 6.0              # long videos: fewer, smoother cuts
    items.sort(key=lambda x: -x[2])
    while len(items) > 1 and total_len(merge_windows([(a, b) for a, b, _ in items], gap)) > max_len + 1e-6:
        items.pop()                                   # drop the weakest moment
    ws = merge_windows([(a, b) for a, b, _ in items], gap)
    if total_len(ws) > max_len + 1e-6:                # one window still too long: keep its middle
        mid = (items[0][0] + items[0][1]) / 2
        a = min(max(lo, mid - max_len / 2), max(lo, hi - max_len))
        ws = [[a, min(hi, a + max_len)]]
    if total_len(ws) < MIN_FILL * max_len:            # too short: add more before and after
        ws = pad_windows(ws, lo, hi, TARGET_FILL * max_len)
    out = [(round(a, 2), round(b, 2)) for a, b in ws]
    over = sum(b - a for a, b in out) - max_len           # rounding must never push us above the limit
    if over > 0 and out and out[-1][1] - out[-1][0] - over >= 3:
        out[-1] = (out[-1][0], round(out[-1][1] - over - 0.005, 2))
    return out


# ---------------- the plan ----------------
def moment_len(max_len):
    """Preferred length of ONE moment: short for Shorts, about 60-80 s for long videos
    (enough time to see the build-up and the reaction, like a person cutting by hand)."""
    return 16.0 if max_len <= 120 else min(75.0, 16.0 + (max_len - 120) * 0.5)


def excitement_curve(db, hop, segments, duration):
    """Excitement for every second: how much louder than usual (crowd, shouting, laughing)
    plus excited / frustrated words and exclamations in the speech."""
    secs = int(duration)
    curve = np.zeros(secs)
    n = len(db)
    if n >= 40:
        sm = np.convolve(db, np.ones(3) / 3, mode="same")
        base = rolling_median(sm, max(5, int(20 / hop)))
        ex = np.clip(sm - base, 0.0, 20.0)
        per = max(1, int(round(1.0 / hop)))
        m = min(secs, n // per)
        if m > 0:
            curve[:m] = ex[: m * per].reshape(m, per).mean(axis=1)
    for s_, e_, t in segments:
        cnt = len(EXCITE_RE.findall(t.lower())) + t.count("!")
        if cnt:
            a, b = int(max(0, s_)), int(min(secs, max(s_ + 1, e_)))
            if b > a:
                curve[a:b] += 1.5 * min(cnt, 4) / (b - a)
    return curve


def pick_moment_windows(curve, lo, hi, L, budget, blocked, min_gap=8.0):
    """Best non-overlapping windows of about L seconds inside [lo, hi], best first, until `budget`
    seconds are chosen. Windows that overlap parts already used (`blocked`) are skipped."""
    n = len(curve)
    lo_i, hi_i = int(max(0, lo)), int(min(n, hi))
    Li = max(8, int(round(L)))
    if hi_i - lo_i < Li:
        return [(float(lo_i), float(hi_i), float(curve[lo_i:hi_i].mean() if hi_i > lo_i else 0.0))] \
            if hi_i - lo_i >= 10 else []
    c = np.concatenate([[0.0], np.cumsum(curve)])
    starts = np.arange(lo_i, hi_i - Li + 1)
    score = (c[starts + Li] - c[starts]) / Li
    for a, b in blocked:
        score[(starts < b + 2) & (starts + Li > a - 2)] = -1.0
    chosen, total = [], 0.0
    while total < budget - 1:
        i = int(np.argmax(score))
        if score[i] < 0:
            break
        st = int(starts[i])
        length = min(float(Li), budget - total)       # the last moment may be shorter
        if chosen and length < 20:
            break
        chosen.append((float(st), st + length, float(score[i])))
        total += length
        score[(starts > st - Li - min_gap) & (starts < st + Li + min_gap)] = -1.0
    return chosen


def build_highlight_plan(analysis, duration, used, mode, max_len, chapters=None):
    """Decide WHICH parts of the video go into the short.
    mode: 'auto' | 'football' | 'general'. `used`: windows already used from this video.
    The result is always in the original video order."""
    db, hop = analysis["db"], analysis["hop"]
    segments, whistles = analysis["segments"], analysis["whistles"]
    if duration <= max_len:
        return {"windows": [(0.0, float(duration))], "mode": "general", "section": (0.0, float(duration)),
                "notes": ["The video is shorter than your limit, so all of it is used."]}

    def fresh(p):
        return not any(a - 2 <= p["peak"] <= b + 2 for a, b in used)

    peaks = [p for p in find_energy_peaks(db, hop) if fresh(p)]                       # normal moments
    fb_all = find_energy_peaks(db, hop, thr=3.5, sep_s=5.0)        # football: more sensitive
    classify_football_peaks(fb_all, segments, whistles)            # goal / foul / other
    fb_peaks = [p for p in fb_all if fresh(p)]
    if not peaks and not fb_peaks:
        raise RuntimeError("No new highlight moments were found in this video "
                           "(everything clear was already used).")
    for p in peaks:
        txt = text_between(segments, p["peak"] - 6, p["peak"] + 4)
        p["score"] = (min(p["excess"], 16) / 8 + 0.3 * min(len(EXCITE_RE.findall(txt)) + txt.count("!"), 3)
                      + 0.2 * min(p["sus"], 5))

    sections = [{"a": a, "b": b} for a, b in segment_topics(segments, duration, chapters)]
    notes = []
    for s in sections:
        mins = max(1.0, (s["b"] - s["a"]) / 60)
        s["fb"] = football_density(segments, s["a"], s["b"])
        s["wh"] = sum(1 for w in whistles if s["a"] <= w["t"] < s["b"])
        s["football"] = s["fb"] >= FB_DENSITY or s["wh"] / mins >= WH_PER_MIN
        s["peaks"] = [p for p in peaks if s["a"] <= p["peak"] < s["b"]]
        s["fbp"] = [p for p in fb_peaks if s["a"] <= p["peak"] < s["b"]]
        s["goals"] = [p for p in s["fbp"] if p["kind"] == "goal"]
        s["fouls"] = [p for p in s["fbp"] if p["kind"] == "foul"]
        s["goals_all"] = [p for p in fb_all if s["a"] <= p["peak"] < s["b"] and p["kind"] == "goal"]
        s["value"] = sum(sorted((p["score"] for p in s["peaks"]), reverse=True)[:5])
    if len(sections) > 1:
        desc = ", ".join(f"{fmt_time(s['a'])}–{fmt_time(s['b'])}" + (" ⚽" if s["football"] else "")
                         for s in sections)
        notes.append(f"🧭 {len(sections)} topic sections found: {desc}")

    fsecs = [s for s in sections if s["football"] and s["goals"]]
    if mode == "football" and not fsecs:
        fsecs = [s for s in sections if s["goals"]]
    use_fb = bool(fsecs) and (mode == "football" or
                              (mode == "auto" and any(len(s["goals"]) >= 2 for s in fsecs)))

    need = TARGET_FILL * max_len

    def bounds(sec):
        """Where the moments may be padded. Normally the chosen section; if that section is shorter than
        the length you asked for, the best neighbouring sections are added until it is long enough."""
        i = next(k for k, x in enumerate(sections) if x is sec)
        l = r = i
        lo, hi = sec["a"], sec["b"]
        while hi - lo < need and (l > 0 or r < len(sections) - 1):
            lv = sections[l - 1]["value"] if l > 0 else -1.0
            rv = sections[r + 1]["value"] if r < len(sections) - 1 else -1.0
            if l > 0 and lv >= rv:
                l -= 1
                lo = sections[l]["a"]
            else:
                r += 1
                hi = sections[r]["b"]
        return lo, hi

    if use_fb:
        sec = max(fsecs, key=lambda s: sum(g["gscore"] for g in s["goals"]))
        matches = split_matches(sec["goals"], whistles, segments, sec["a"], sec["b"])
        best = max(matches, key=lambda m: (len(m), sum(g["gscore"] for g in m)))
        blo, bhi = bounds(sec)
        windows = fit_windows(goal_windows(best, max_len), blo, bhi, max_len)
        if (blo, bhi) != (sec["a"], sec["b"]):
            notes.append("ℹ️ The football section was shorter than your length, so the parts around it "
                         "were included to reach it.")
        lo, hi = best[0]["peak"] - 60, best[-1]["peak"] + 60
        n_f = sum(1 for p in sec["fouls"] if lo <= p["peak"] <= hi)
        notes.append(f"🔔 {sum(1 for w in whistles if sec['a'] <= w['t'] < sec['b'])} whistles heard in the "
                     f"football section · {len(matches)} match(es) found")
        notes.append(f"⚽ Using the match with {len(best)} goal moment(s), in order, at "
                     + ", ".join(fmt_time(g["peak"]) for g in best)
                     + f" · {n_f} whistle/foul moment(s) skipped")
        return {"windows": windows, "mode": "football", "section": (sec["a"], sec["b"]), "notes": notes}

    if used and mode != "general" and any(s["football"] and len(s["goals_all"]) >= 2 for s in sections):
        raise RuntimeError("All the goal moments of this football video were already used. "
                           "Pick another video (or choose 'Best moments' to cut other parts).")

    def value(s):
        return sum(sorted((p["score"] for p in s["peaks"]), reverse=True)[:5])

    sec = max(sections, key=value)
    blo, bhi = bounds(sec)                            # normally the section; wider if it is too short
    L = moment_len(max_len)
    curve = excitement_curve(db, hop, segments, duration)
    picks = pick_moment_windows(curve, blo, bhi, L, need, used)
    if not picks:
        raise RuntimeError("Couldn't find clear highlight moments in this video.")
    windows = fit_windows([(a, b, v) for a, b, v in picks], blo, bhi, max_len)
    if (blo, bhi) != (sec["a"], sec["b"]):
        notes.append("ℹ️ The best section was shorter than your length, so the parts around it were included "
                     "to reach it.")
    items = picks
    notes.append(f"🔥 Picked {len(items)} exciting moment(s) of about {L:.0f}s each from "
                 f"{fmt_time(blo)}–{fmt_time(bhi)}, kept in video order")
    return {"windows": windows, "mode": "general", "section": (sec["a"], sec["b"]), "notes": notes}


# ============================================================================
# ONE-CLIP MODE: the best single moment (YouTube "most replayed", else the loudest audio)
# ============================================================================
MAX_MOMENTS = 5      # at most this many different moments are ranked per video
MIN_RATIO = 0.4      # a moment must have >= 40% of the best moment's score


def rank_windows(scores, length, max_n=MAX_MOMENTS, min_ratio=MIN_RATIO):
    """All distinct (non-overlapping) windows, best first: [(start_second, score), ...]"""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return [(0, 0.0)]
    length = max(1, min(int(length), n))
    c = np.concatenate([[0.0], np.cumsum(scores)])
    means = (c[length:] - c[:-length]) / length if length < n else np.array([c[-1] / n])
    work = means.copy()
    best = float(work.max())
    out = []
    while len(out) < max_n:
        i = int(np.argmax(work))
        v = float(work[i])
        if v < 0 or (out and (v <= 0 or v < best * min_ratio)):
            break
        out.append((i, v))
        work[max(0, i - length + 1): i + length] = -1.0     # block overlapping windows
    return out


def heatmap_ranking(info, length):
    """Ranked moments from YouTube's 'Most replayed' heatmap, or None if the video has none."""
    duration = int(info.get("duration") or 0)
    heat = info.get("heatmap")
    if not heat or duration <= 0:
        return None
    per_sec = np.zeros(duration + 1)
    for seg in heat:
        a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
        per_sec[a:b + 1] = seg["value"]
    return rank_windows(per_sec[:duration], length)


def audio_energy_ranking(path, total, length):
    """No heatmap: ranked moments by loudness of the (small, audio-only) download."""
    parts = []
    for c0 in range(0, int(total), CHUNK_SECONDS):
        clen = int(min(CHUNK_SECONDS, total - c0))
        pcm = decode_mono16k(path, c0, clen)
        secs = len(pcm) // ANALYSIS_SR
        if secs == 0:
            parts.append(np.zeros(clen))
            continue
        r = np.sqrt((pcm[: secs * ANALYSIS_SR].reshape(secs, ANALYSIS_SR) ** 2).mean(axis=1))
        parts.append(np.pad(r, (0, max(0, clen - secs)))[:clen])
    return rank_windows(np.concatenate(parts) if parts else np.zeros(1), length)


def choose_single(entry, ranked, length):
    """Best moment of this video that does not overlap anything already used.
    Returns (start, number_of_this_moment, total_moments)."""
    used = used_windows(entry)
    if entry.get("legacy") and not used and ranked:
        used = [(ranked[0][0], ranked[0][0] + length)]       # old versions used the top moment
    for rank, (start, _score) in enumerate(ranked, 1):
        if all(start + length <= a or start >= b for a, b in used):
            return start, rank, len(ranked)
    raise RuntimeError(
        f"You've already made clips from all {len(ranked)} viral moment(s) of this video. "
        "Pick another video, or choose 'Multiple moments' to cut other parts of it.")


# ---------------- download each moment, join them ----------------
def download_audio(info, workdir):
    return _download_try(info, workdir, None, name="audio", fmt=AUDIO_FMT)


def concat_segments(files, out_path):
    """files: [(path, offset_in_file, length)] -> one mp4 (all moments one after another)."""
    n = len(files)
    base = [FFMPEG, "-y", "-v", "error"]
    for path, off, length in files:
        base += ["-ss", f"{off:.3f}", "-t", f"{length:.3f}", "-i", path]
    enc_v = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    vparts = "".join(f"[{i}:v]setsar=1,fps={FPS},format=yuv420p[v{i}];" for i in range(n))
    aparts = "".join(f"[{i}:a]aresample={SR},asetpts=PTS-STARTPTS[a{i}];" for i in range(n))
    both = "".join(f"[v{i}][a{i}]" for i in range(n))
    try:
        subprocess.run(base + ["-filter_complex", vparts + aparts + both + f"concat=n={n}:v=1:a=1[v][a]",
                               "-map", "[v]", "-map", "[a]"] + enc_v + ["-c:a", "aac", "-b:a", "192k", out_path],
                       check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:        # a moment without audio -> video only
        vonly = "".join(f"[v{i}]" for i in range(n))
        subprocess.run(base + ["-filter_complex", vparts + vonly + f"concat=n={n}:v=1:a=0[v]",
                               "-map", "[v]", "-an"] + enc_v + [out_path],
                       check=True, stderr=subprocess.DEVNULL)


def group_spans(windows, total, pad=3.0, bridge=30.0, max_spans=6, max_span_len=1500.0):
    """Few big downloads instead of one request per moment (YouTube blocks many quick requests).
    Moments that are close together share one download; every moment lies inside exactly one span."""
    spans = sorted([max(0.0, a - pad), min(float(total), b + pad)] for a, b in windows)
    merged = []
    for sp in spans:
        if merged and sp[0] - merged[-1][1] <= bridge and sp[1] - merged[-1][0] <= max_span_len:
            merged[-1][1] = max(merged[-1][1], sp[1])
        else:
            merged.append(sp)
    while len(merged) > max_spans:                      # still many: join the closest neighbours
        gaps = [(merged[i + 1][0] - merged[i][1], i) for i in range(len(merged) - 1)
                if merged[i + 1][1] - merged[i][0] <= max_span_len]
        if not gaps:
            break
        _, i = min(gaps)
        merged[i][1] = merged[i + 1][1]
        del merged[i + 1]
    return merged


def download_span(info, workdir, span, name, fmt, tries=2):
    """One download piece; returns the file, or None if YouTube blocked it."""
    for k in range(tries):
        try:
            return _download_try(info, workdir, span, name=name, fmt=fmt)
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def load_user_sound(path):
    """A transition sound YOU chose (mp3/wav...). Returns stereo audio or None."""
    try:
        a = decode_stereo(path, 0, 1.5)[: int(1.5 * SR)]
        peak = float(np.abs(a).max())
        if peak < 1e-4:
            return None
        a = a / peak
        n = int(0.15 * SR)
        a[-n:] *= np.linspace(1, 0, n, dtype=np.float32)[:, None]
        return a.astype(np.float32)
    except Exception:
        return None


def save_to_downloads(path):
    """Copy the finished video straight into ~/Downloads (so a page reload never loses it)."""
    try:
        folder = os.path.expanduser("~/Downloads")
        os.makedirs(folder, exist_ok=True)
        dest = unique_path(folder, os.path.splitext(os.path.basename(path))[0])
        shutil.copy2(path, dest)
        return dest
    except Exception:
        return None


def used_windows(entry):
    w = [tuple(x) for x in (entry.get("windows") or [])]
    if not w:                                     # clips made by older versions
        w = [(m, m + 90) for m in (entry.get("moments") or [])]
    return w


def record_use_many(info, windows):
    h = load_history()
    e = h.get(info["id"]) or new_entry(info)
    e["windows"] = [list(x) for x in (e.get("windows") or [])] + [[round(a, 1), round(b, 1)] for a, b in windows]
    e["moments"] = list(e.get("moments") or []) + [int(a) for a, _ in windows]
    e["legacy"] = False
    e["last_used"] = dt.datetime.now().isoformat(timespec="seconds")
    h[info["id"]] = e
    save_history(h)


# ---------------- glitch transition ----------------
def glitch_frame(frame, amount, seed):
    """RGB split + sliding bands + scan noise. amount 0..1. Works for any frame size."""
    rng = np.random.default_rng(seed)
    fh, fw = frame.shape[:2]
    kh, kw = fh / 1920.0, fw / 1080.0                       # scale the effect to the frame
    f = frame.copy()
    for _ in range(int(2 + 8 * amount)):
        h = max(6, int(rng.integers(12, 140) * max(kh, 0.5)))
        y = int(rng.integers(0, max(1, fh - h)))
        shift = int(rng.choice([-1, 1])) * max(8, int(rng.integers(30, 60 + int(240 * amount)) * kw))
        f[y:y + h] = np.roll(f[y:y + h], shift, axis=1)
    d = max(3, int((6 + 34 * amount) * kw))
    out = f.copy()
    out[:, :, 0] = np.roll(f[:, :, 0], d, axis=1)
    out[:, :, 2] = np.roll(f[:, :, 2], -d, axis=1)
    rows = rng.integers(0, fh, size=int(10 + 30 * amount))
    out[rows] = rng.integers(0, 255, size=(len(rows), fw, 3), dtype=np.uint8)
    if amount > 0.85:
        out = np.clip(out.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    return out


def dip_audio(track, pos, half=350):
    """Tiny volume dip so a hard audio cut doesn't click."""
    a, b = max(0, pos - half), min(len(track), pos + half)
    if b - a < 4:
        return
    w = np.abs(np.linspace(-1, 1, b - a))[:, None].astype(np.float32)
    track[a:b] *= (0.15 + 0.85 * w)


# ---------------- final render: pop captions + glitch transitions ----------------
def probe_size(path):
    """(width, height) of the video, or (None, None)."""
    try:
        r = subprocess.run([FFMPEG, "-hide_banner", "-i", path], stderr=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, text=True)
        m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", r.stderr)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)
    except Exception:
        return (None, None)


def render_final(src, dur, words, out_path, fx, cuts=None, start=0.0, fmt="short", sound=None):
    """Final video: 'short' = 9:16 with blurred background, 'regular' = 16:9 HD.
    Captions that pop, glitch transitions at each join."""
    fx = {**FX_DEFAULTS, **(fx or {})}
    st = FORMATS[fmt]
    w, h = st["w"], st["h"]
    total = int(dur * FPS)
    cut_frames = [int(round(c * FPS)) for c in (cuts or [])]

    track = normalize_audio(decode_stereo(src, start, dur)[: (total + 1) * SPF])      # even volume
    for cf in cut_frames:
        dip_audio(track, cf * SPF)
    if fx["glitch"] and cut_frames and sound:          # only a sound YOU chose; there is no built-in sound
        gl = load_user_sound(sound)
        if gl is not None:
            for cf in cut_frames:
                add_sfx(track, gl, max(0, cf - 2), 0.28)
    wav_path = os.path.splitext(out_path)[0] + "_audio.wav"
    write_wav(wav_path, soft_limit(track))

    groups = make_groups(words, size=st["words"])
    sw, sh = probe_size(src)
    if fmt == "regular":
        if sw and sh and abs(sw / sh - w / h) < 0.02:          # already 16:9: just scale
            vf = f"scale={w}:{h},fps={FPS}"
        else:                                                   # other shapes: fit + blurred background
            vf = (f"split[a][b];"
                  f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bg];"
                  f"[b]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={FPS}")
    else:
        vf = (f"split[a][b];"
              f"[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bg];"
              f"[b]scale={w}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={FPS}")
    dec = subprocess.Popen(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src, "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
         "-i", wav_path, "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    fsize = w * h * 3
    gi, j = 0, 0
    while True:
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        t = j / FPS
        if fx["glitch"] and cut_frames:
            amt = max((1 - abs(j - cf) / 5.0 for cf in cut_frames if abs(j - cf) < 5), default=0.0)
            if amt > 0:
                frame = glitch_frame(frame, amt, j)
        if groups:
            while gi + 1 < len(groups) and groups[gi + 1]["start"] <= t:
                gi += 1
            g = groups[gi]
            if g["start"] <= t < g["end"]:
                active = max(k for k, wd in enumerate(g["words"]) if wd[1] <= t)
                pop = 1.0
                if fx["pop"]:
                    age = t - g["words"][active][1]
                    pop = 1.22 if age < 0.06 else (1.12 if age < 0.12 else 1.0)
                rgb, alpha = caption_strip([wd[0] for wd in g["words"]], active, pop, st)
                blend(frame, rgb, alpha, st["strip_y"])
        enc.stdin.write(frame.tobytes())
        j += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    try:
        os.remove(wav_path)
    except OSError:
        pass
    n = len(cut_frames)
    return (f"{n} glitch transition{'s' if n != 1 else ''}" if fx["glitch"] and n else "no transitions (single clip)")


def clean_filename(title, fallback="video"):
    """Turn a YouTube title into a safe file name (same name as the video)."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", title or "")
    name = re.sub("[\U00010000-\U0010ffff]", "", name)   # drop emoji
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:120].strip(" .") or fallback


def unique_path(folder, name, ext=".mp4"):
    path = os.path.join(folder, name + ext)
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name} ({n}){ext}")
        n += 1
    return path


def parse_time(s):
    """'12:12', '1:02:03' or '90' -> seconds."""
    parts = s.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p.strip()) for p in parts):
        raise ValueError(s)
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def parse_timestamps(text, default_len, total):
    """'12:12-15:09, 16:00-16:54, 20:30' -> [(start, end), ...] in the order typed.
    A single time starts a clip there and runs for default_len seconds."""
    windows = []
    for item in [t.strip() for t in re.split(r"[,;\n]+", text or "") if t.strip()]:
        halves = re.split(r"\s*(?:[-\u2013\u2014]|\bto\b)\s*", item)
        try:
            if len(halves) == 1:
                a = parse_time(halves[0])
                b = a + default_len
            elif len(halves) == 2:
                a, b = parse_time(halves[0]), parse_time(halves[1])
            else:
                raise ValueError(item)
        except ValueError:
            raise RuntimeError(f"Couldn't read \u201c{item}\u201d. Use 12:12-15:09 for a range, or just 12:12.")
        if a >= total:
            raise RuntimeError(f"{item}: that's past the end of the video ({fmt_time(total)}).")
        b = min(b, float(total))
        if b - a < 1:
            raise RuntimeError(f"{item}: the end has to come after the start.")
        windows.append((a, b))
    return windows


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
FORMAT_CHOICES = ["Short (vertical, up to 3 min)", "Regular video (horizontal HD, up to 10 min)"]
CLIP_TYPES = ["One clip", "Multiple moments joined"]
MODE_CHOICES = ["Auto-detect", "Football goals (one match)", "Best moments (one topic)"]


def run(query, link, length, cc_only, min_views, own_channel, video_format, clip_type, mode_label, sound_path, stamps, *fx_args):
    query = (query or "").strip()
    link = (link or "").strip()
    if not query and not link:
        yield "Type a topic/title OR paste a YouTube link (at least one is needed).", None, None, ""
        return
    multi = str(clip_type).startswith("Multiple")
    mode = {"Football goals (one match)": "football",
            "Best moments (one topic)": "general"}.get(mode_label, "auto")
    regular = str(video_format).startswith("Regular")
    fmt = "regular" if regular else "short"
    src_fmt = VIDEO_FMT_HD if regular else VIDEO_FMT          # regular videos are downloaded in HD
    max_len = int(length)
    cap_note = ""
    if max_len > FORMATS[fmt]["max_len"]:
        max_len = FORMATS[fmt]["max_len"]
        cap_note = f"  \n⚠️ This format is limited to {max_len} seconds, so {max_len}s was used."
    workdir = tempfile.mkdtemp(prefix="viralclip_")
    try:
        if link:
            yield "🔗 Opening your video link…", None, None, ""
            info = use_link(link)
        else:
            yield f"🔎 Searching YouTube for “{query}” (last 2 years)…", None, None, ""
            info = find_video(query, cc_only, min_views, own_channel)
        meta = (f"**{info['title']}**  \n{info.get('channel') or info.get('uploader')} · "
                f"{info.get('view_count', 0):,} views · uploaded {info['upload_date']}  \n"
                f"{info['webpage_url']}") + cap_note

        total = int(info["duration"])
        entry = get_entry(info)
        manual = parse_timestamps(stamps, max_len, total) if (stamps or "").strip() else []
        if manual:
            cap = FORMATS[fmt]["max_len"]
            tot = sum(b - a for a, b in manual)
            if tot > cap + 0.5:
                raise RuntimeError(f"Your timestamps add up to {fmt_time(tot)}, but this format allows {fmt_time(cap)}. "
                                   "Shorten them or switch the video format.")
        fx = {k: v for k, v in zip(FX_KEYS, fx_args) if v is not None}
        outdir = tempfile.mkdtemp(prefix="viralclip_out_")      # temporary, not your folders
        out = unique_path(outdir, clean_filename(info.get("title"), info.get("id")))

        if not multi and not manual:
            # ======================= ONE CLIP: best single moment, no transitions =======================
            ranked = heatmap_ranking(info, max_len)
            if ranked is not None:
                how = "YouTube 'Most replayed' heatmap"
            else:
                how = "loudest audio"
                yield "⬇️ No 'Most replayed' data: downloading the audio to find the loudest moment…", None, None, meta
                audio_path = download_audio(info, workdir)
                yield "🔥 Finding the most viral moment…", None, None, meta
                ranked = audio_energy_ranking(audio_path, total, max_len)
            start, nth, count = choose_single(entry, ranked, max_len)
            dur = float(min(max_len, total - start))
            yield "⬇️ Downloading just the best part of the video…", None, None, meta
            src, offset = download(info, workdir, (max(0.0, start - 3.0), min(float(total), start + dur + 3.0)), fmt=src_fmt)
            clip_start = start - offset
            meta += (f"  \nMoment {nth} of {count} found · {fmt_time(start)}–{fmt_time(start + dur)} · via {how}  \n"
                     f"▶️ Check the source moment: https://www.youtube.com/watch?v={info['id']}&t={int(start)}s")

            yield "🎙️ Transcribing speech (first run downloads the model)…", None, None, meta
            words = transcribe(src, clip_start, dur, workdir)
            yield "🎬 Rendering with captions that pop (a few minutes)…", None, None, meta
            summary = render_final(src, dur, words, out, fx, [], start=clip_start, fmt=fmt)
            meta += f"  \n✨ Edits: captions that pop · {summary}"
            record_use_many(info, [(float(start), float(start) + dur)])
        else:
            # ======================= MULTIPLE MOMENTS: joined, glitch transitions =======================
            if manual:
                windows = manual
            else:
                used = used_windows(entry)           # parts of this video that were already used
                yield "⬇️ Downloading the audio of the full video (small file)…", None, None, meta
                audio_path = download_audio(info, workdir)
                gen = analyze_video(audio_path, total)
                while True:                           # forward progress text, grab the result
                    try:
                        msg = next(gen)
                    except StopIteration as stop:
                        analysis = stop.value
                        break
                    yield msg, None, None, meta
                analyzed = min(total, analysis["analyzed"])
                if analyzed < total:
                    meta += f"  \n⚠️ Long video: the first {fmt_time(analyzed)} were analysed."

                yield "🧠 Finding sections, goals and the best moments…", None, None, meta
                plan = build_highlight_plan(analysis, analyzed, used, mode, max_len, info.get("chapters"))
                windows = plan["windows"]
                if not windows:
                    raise RuntimeError("No usable highlight moments were found in this video.")
                meta += "".join(f"  \n{n}" for n in plan["notes"])

            # download in a few big pieces (not one request per moment), then cut the moments out
            planned = float(sum(b - a for a, b in windows))
            spans = group_spans(windows, total)
            span_files = []
            for si, sp in enumerate(spans):
                yield (f"⬇️ Downloading part {si + 1} of {len(spans)} "
                       f"({fmt_time(sp[0])}–{fmt_time(sp[1])})…"), None, None, meta
                span_files.append(download_span(info, workdir, (sp[0], sp[1]), f"seg{si:02d}", src_fmt))
            files, kept, failed = [], [], 0
            for a, b in windows:
                si = next((k for k, sp in enumerate(spans) if sp[0] - 1e-6 <= a and b <= sp[1] + 1e-6), None)
                if si is None or span_files[si] is None:
                    failed += 1
                    continue
                files.append((span_files[si], a - spans[si][0], b - a))
                kept.append((a, b))
            if not files:
                raise RuntimeError("Couldn't download any of the moments (YouTube blocked it). "
                                   "Wait a few minutes and try again.")
            got = float(sum(l for _, _, l in files))
            if got < 0.85 * planned:
                meta += (f"  \n⚠️ Only {got / 60:.1f} of the planned {planned / 60:.1f} minutes could be downloaded "
                         f"(YouTube blocked {failed} moment(s)). Try again in a few minutes.")

            yield "🎞️ Joining the moments…", None, None, meta
            combo = os.path.join(workdir, "combo.mp4")
            concat_segments(files, combo)
            lens = [length_ for _, _, length_ in files]
            cuts = list(np.cumsum(lens)[:-1])
            combo_len = float(sum(lens))
            clips = " · ".join(f"{fmt_time(a)}–{fmt_time(b)}" for a, b in kept)
            limit = "" if manual else f", limit {max_len}s"
            meta += f"  \n🎬 Clips used ({combo_len:.0f}s total{limit}): {clips}"
            meta += (f"  \n▶️ Check the first moment: "
                     f"https://www.youtube.com/watch?v={info['id']}&t={int(kept[0][0])}s")

            yield "🎙️ Transcribing speech for captions…", None, None, meta
            words = transcribe(combo, 0, combo_len, workdir)
            yield "🎬 Rendering with captions and glitch transitions (a few minutes)…", None, None, meta
            summary = render_final(combo, combo_len, words, out, fx, cuts, fmt=fmt, sound=sound_path)
            meta += f"  \n✨ Edits: captions that pop · {summary}"
            record_use_many(info, kept)          # shows up in "Last used"; next time = new moments

        saved = save_to_downloads(out)
        if saved:
            meta += f"  \n💾 Saved to your Downloads folder: {saved}"
            yield f"✅ Done! Saved to your Downloads folder: {os.path.basename(saved)}", out, out, meta
        else:
            yield ("✅ Done! (Couldn't save to the Downloads folder automatically; "
                   "use the Download button.)"), out, out, meta
    except Exception as e:
        yield f"❌ Error: {e}", None, None, ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cleanup_old_outputs(max_age_hours=24):
    import glob, time
    for p in glob.glob(os.path.join(tempfile.gettempdir(), "viralclip_out_*")):
        try:
            if time.time() - os.path.getmtime(p) > max_age_hours * 3600:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


cleanup_old_outputs()

CP_CSS = r'''@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root:root,.dark.dark,.gradio-container.gradio-container{
 --cp-bg:#07070b;--cp-card:rgba(255,255,255,.035);--cp-line:rgba(255,255,255,.09);
 --cp-text:#f3f3f7;--cp-mute:#9494a8;--cp-accent:#8f6bff;--cp-pop:#ffb224;--cp-violet:#7b5cff;--cp-pink:#ff4fa3;
 --body-background-fill:#07070b;--background-fill-primary:#07070b;--background-fill-secondary:#0e0e15;
 --block-background-fill:transparent;--block-border-width:0px;--block-shadow:none;
 --block-label-background-fill:transparent;--block-label-text-color:#9494a8;--block-title-text-color:#9494a8;
 --body-text-color:#f3f3f7;--body-text-color-subdued:#9494a8;--border-color-primary:rgba(255,255,255,.09);
 --input-background-fill:#10101a;--input-border-color:rgba(255,255,255,.09);--input-border-color-focus:#8f6bff;
 --input-shadow-focus:0 0 0 3px rgba(143,107,255,.15);--input-placeholder-color:#5d5d70;
 --color-accent:#8f6bff;--color-accent-soft:rgba(143,107,255,.12);--slider-color:#8f6bff;
 --checkbox-background-color:#10101a;--checkbox-background-color-selected:#8f6bff;--checkbox-border-color-selected:#8f6bff;
 --button-primary-background-fill:#8f6bff;--button-primary-text-color:#ffffff;
 --button-secondary-background-fill:rgba(255,255,255,.06);--button-secondary-text-color:#f3f3f7;
 --button-secondary-border-color:rgba(255,255,255,.09);--radius-lg:16px;--radius-md:12px;
}
html{background:#07070b!important;color-scheme:dark;scroll-behavior:smooth}
html,body{width:100%;max-width:100%}
html{overflow-x:hidden}
body{overflow-x:clip}
.gradio-container img,.gradio-container video{max-width:100%}
body{background:transparent!important;font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif}
body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
 background:radial-gradient(60rem 40rem at 12% -10%,rgba(123,92,255,.24),transparent 60%),
 radial-gradient(50rem 36rem at 98% 4%,rgba(178,102,255,.14),transparent 60%)}
gradio-app,.gradio-container{background:transparent!important}
gradio-app{width:100%!important}
.gradio-container{width:100%!important;min-width:0!important;max-width:min(960px,100%)!important;box-sizing:border-box!important;margin:0 auto!important;padding:0 20px 120px!important}
footer{display:none!important}
.cp-chrome{display:none!important}

/* ---------- fixed chrome ---------- */
#cp-progress{position:fixed;top:0;left:0;right:0;height:3px;z-index:101;transform-origin:0 50%;transform:scaleX(0);
 background:linear-gradient(90deg,#8f6bff,#ffb224)}
#cp-nav{position:fixed;top:calc(12px + env(safe-area-inset-top,0px));left:50%;transform:translateX(-50%);z-index:100;display:flex;align-items:center;gap:6px;width:max-content;
 padding:7px 8px 7px 18px;border-radius:999px;background:rgba(14,14,21,.55);border:1px solid rgba(255,255,255,.09);
 -webkit-backdrop-filter:blur(16px) saturate(1.5);backdrop-filter:blur(16px) saturate(1.5);max-width:calc(100vw - 24px);
 transition:background .3s,box-shadow .3s}
body.cp-scrolled #cp-nav{background:rgba(10,10,16,.86);box-shadow:0 12px 40px rgba(0,0,0,.55)}
#cp-nav .brand{font:700 15px 'Space Grotesk',sans-serif;color:#f3f3f7;margin-right:8px;white-space:nowrap;letter-spacing:-.01em}
#cp-nav .brand{cursor:pointer;transition:opacity .2s,transform .15s}
#cp-nav .brand:hover{opacity:.8}
#cp-nav .brand:active{transform:scale(.96)}
#cp-nav .brand i{color:#ffb224;text-shadow:0 0 12px rgba(255,178,36,.6);font-style:normal;margin-right:6px}
#cp-nav .links{position:relative;display:flex;gap:2px;min-width:0;overflow-x:auto;scrollbar-width:none;-webkit-overflow-scrolling:touch}
#cp-nav .links::-webkit-scrollbar{display:none}
#cp-nav button{all:unset;box-sizing:border-box;cursor:pointer;padding:8px 14px;border-radius:999px;color:#9494a8;font:500 13px 'Inter',sans-serif;
 white-space:nowrap;transition:color .2s,background .2s,transform .15s}
#cp-nav button:hover{color:#f3f3f7;background:rgba(255,255,255,.07)}
#cp-nav button:active{transform:scale(.94)}
#cp-nav button.active{color:#fff;background:#8f6bff;box-shadow:0 4px 18px rgba(143,107,255,.5)}
#cp-fab{all:unset;box-sizing:border-box;position:fixed;right:max(16px,env(safe-area-inset-right,0px));bottom:max(16px,env(safe-area-inset-bottom,0px));z-index:100;width:48px;height:48px;border-radius:50%;
 display:grid;place-items:center;cursor:pointer;font-size:20px;color:#fff;background:#8f6bff;box-shadow:0 10px 30px rgba(143,107,255,.5);
 opacity:0;pointer-events:none;transform:translateY(14px) scale(.8);transition:opacity .3s,transform .3s}
#cp-fab.show{opacity:1;pointer-events:auto;transform:none}
#cp-fab:hover{transform:translateY(-3px)}
#cp-fab:active{transform:scale(.9)}

/* ---------- hero ---------- */
.cp-hero{position:relative;padding:118px 0 30px;text-align:center}
.cp-hero-inner{will-change:transform,opacity}
.cp-pill{display:inline-flex;align-items:center;gap:9px;padding:7px 15px;border:1px solid rgba(255,255,255,.09);border-radius:999px;
 background:rgba(255,255,255,.035);color:#9494a8;font-size:13px}
.cp-pill b{width:7px;height:7px;border-radius:50%;background:#ffb224;box-shadow:0 0 12px #ffb224;animation:cp-pulse 1.8s ease-in-out infinite}
.gradio-container .cp-hero h1{font:700 clamp(2.6rem,7.4vw,5.2rem)/1.02 'Space Grotesk',sans-serif;letter-spacing:-.035em;margin:22px 0 16px;color:#f3f3f7;text-wrap:balance}
.cp-grad{background:linear-gradient(100deg,#c4a8ff,#8f6bff 55%,#b266ff);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;
 color:transparent;animation:cp-shift 6s ease-in-out infinite alternate}
.gradio-container .cp-hero p{color:#9494a8;font-size:1.12rem;margin:0 auto 28px;max-width:32rem;text-wrap:balance}
.cp-cta{all:unset;box-sizing:border-box;cursor:pointer;position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:10px;padding:15px 28px;
 border-radius:999px;background:#f3f3f7;color:#0b0b0f;font:600 15px 'Inter',sans-serif;transition:transform .2s,box-shadow .3s}
.cp-cta:hover{transform:translateY(-3px);box-shadow:0 14px 40px rgba(243,243,247,.2)}
.cp-cta:active{transform:scale(.96)}
.cp-cta span{display:inline-block;transition:transform .25s}
.cp-cta:hover span{transform:translateY(3px)}

.cp-tl{margin:44px auto 0;max-width:640px;text-align:left}
.cp-tl-track{position:relative;height:54px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);overflow:hidden;
 background-image:repeating-linear-gradient(90deg,transparent 0 31px,rgba(255,255,255,.05) 31px 32px)}
.cp-seg{position:absolute;top:7px;bottom:7px;border-radius:9px;background:linear-gradient(135deg,#8f6bff,#c084fc);opacity:.3}
.cp-seg b{position:absolute;left:8px;bottom:5px;font:600 10px 'Inter',sans-serif;color:#fff;white-space:nowrap}
.cp-seg.s1{left:12%;width:24%;animation:cp-lit1 9s linear infinite}
.cp-seg.s2{left:46%;width:14%;animation:cp-lit2 9s linear infinite}
.cp-seg.s3{left:72%;width:16%;animation:cp-lit3 9s linear infinite}
.cp-head{position:absolute;top:0;bottom:0;width:2px;background:#ffb224;box-shadow:0 0 14px #ffb224;animation:cp-play 9s linear infinite}
.cp-tl-cap{display:flex;justify-content:space-between;margin-top:9px;color:#5d5d70;font-size:12px}

.cp-marquee{position:relative;height:2rem;margin:34px 0 0;overflow:hidden;-webkit-mask-image:linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent);
 mask-image:linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent)}
.cp-track{position:absolute;top:0;left:0;display:flex;gap:2.2rem;width:max-content;animation:cp-scroll 30s linear infinite}
.cp-marquee:hover .cp-track{animation-play-state:paused}
.cp-track span{font:700 1.05rem 'Space Grotesk',sans-serif;letter-spacing:.16em;text-transform:uppercase;color:rgba(255,255,255,.26);white-space:nowrap;transition:color .2s}
.cp-track span:hover{color:#b79cff}
.cp-track span::after{content:"\2726";margin-left:2.2rem;color:#ffb224;opacity:.85}

/* ---------- steps ---------- */
.gradio-container .step{position:relative;isolation:isolate;margin-top:26px!important;padding:26px!important;gap:16px!important;
 border:1px solid rgba(255,255,255,.09)!important;border-radius:24px!important;background:rgba(255,255,255,.035)!important;
 transition:border-color .3s,opacity .7s cubic-bezier(.2,.7,.2,1),transform .7s cubic-bezier(.2,.7,.2,1)}
.step::before{content:"";position:absolute;inset:0;z-index:-1;border-radius:inherit;pointer-events:none;opacity:0;transition:opacity .3s;
 background:radial-gradient(420px circle at var(--mx,50%) var(--my,0%),rgba(143,107,255,.10),transparent 60%)}
.step:hover::before{opacity:1}
.step:hover{border-color:rgba(143,107,255,.28)!important}
body.cp-js .step,body.cp-js .reveal,body.cp-js .recent-card{opacity:0;transform:translateY(30px)}
body.cp-js .step.in,body.cp-js .reveal.in,body.cp-js .recent-card.in{opacity:1;transform:none}
.step-head{display:flex;gap:16px;align-items:center}
.step-no{font:700 2.8rem/1 'Space Grotesk',sans-serif;background:linear-gradient(180deg,rgba(255,255,255,.55),rgba(255,255,255,.04));
 -webkit-background-clip:text;background-clip:text;color:transparent}
.gradio-container .step-head h2{margin:0;font:700 1.45rem 'Space Grotesk',sans-serif;letter-spacing:-.01em;color:#f3f3f7}
.gradio-container .step-head p{margin:3px 0 0;color:#9494a8;font-size:.92rem}

/* ---------- inputs ---------- */
.gradio-container input[type=text],.gradio-container input[type=number],.gradio-container textarea{border-radius:14px!important;transition:box-shadow .2s,border-color .2s}
.gradio-container button,.card-radio label,.chip label{position:relative;overflow:hidden}
.card-radio .wrap,.card-radio [role=radiogroup]{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(min(230px,100%),1fr));gap:12px!important}
.card-radio label{display:flex!important;align-items:center;gap:14px;min-height:76px;padding:16px 18px!important;cursor:pointer;font-weight:500;
 border:1.5px solid rgba(255,255,255,.09)!important;border-radius:16px!important;background:rgba(255,255,255,.03)!important;
 transition:transform .2s,border-color .2s,background .2s,box-shadow .2s}
.card-radio label:hover{transform:translateY(-3px);border-color:rgba(255,255,255,.25)!important}
.card-radio label:active{transform:scale(.97)}
.card-radio label:has(input:checked),.card-radio label.selected{border-color:#8f6bff!important;background:rgba(143,107,255,.07)!important;
 box-shadow:0 0 0 4px rgba(143,107,255,.08),0 12px 30px rgba(0,0,0,.35)}
.card-radio input{position:absolute!important;opacity:0!important;pointer-events:none}
.card-radio label span{white-space:normal!important;min-width:0}
.fmt-cards label::before{content:"";flex:none;border:2px solid currentColor;border-radius:6px;opacity:.75}
.fmt-cards label:nth-of-type(1)::before{width:20px;height:34px}
.fmt-cards label:nth-of-type(2)::before{width:38px;height:24px}
.type-cards label::before{content:"";flex:none;width:10px;height:22px;border-radius:3px;background:currentColor;opacity:.8}
.type-cards label:nth-of-type(2)::before{margin-right:34px;box-shadow:16px 0 0 currentColor,32px 0 0 currentColor}
.chip{min-width:0!important;flex:0 0 auto!important}
.chip label{display:inline-flex!important;align-items:center;gap:8px;padding:10px 18px!important;cursor:pointer;user-select:none;
 border:1.5px solid rgba(255,255,255,.09)!important;border-radius:999px!important;background:rgba(255,255,255,.03)!important;
 transition:transform .2s,border-color .2s,background .2s,color .2s}
.chip label:hover{transform:translateY(-2px)}
.chip label:active{transform:scale(.95)}
.chip label:has(input:checked){border-color:#8f6bff!important;background:rgba(143,107,255,.16)!important;color:#d4c4ff}
.chip label:has(input:checked)::before{content:"\2713";font-weight:700;color:#ffb224}
.chip input{display:none!important}
.gradio-container button.cp-go{width:100%;min-height:64px;border:0!important;border-radius:18px!important;color:#fff!important;
 font:700 1.15rem 'Space Grotesk',sans-serif!important;background:linear-gradient(100deg,#7c5cff,#b266ff)!important;
 box-shadow:0 10px 40px rgba(143,107,255,.25);transition:transform .2s,box-shadow .3s}
.cp-go:hover{transform:translateY(-2px);box-shadow:0 16px 54px rgba(143,107,255,.4)}
.cp-go:active{transform:scale(.98)}
.cp-go::after{content:"";position:absolute;inset:0;pointer-events:none;transform:translateX(-120%);
 background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.55) 50%,transparent 70%)}
.cp-go:hover::after{transform:translateX(120%);transition:transform .8s}
.gradio-container button.cp-ghost{border-radius:999px!important;border:1px solid rgba(255,255,255,.12)!important;background:rgba(255,255,255,.05)!important;
 transition:transform .2s,background .2s,color .2s}
.cp-ghost:hover{background:#8f6bff!important;color:#fff!important;transform:translateY(-2px)}
.cp-ghost:active{transform:scale(.95)}
.cp-status textarea{font-family:ui-monospace,SFMono-Regular,Menlo,monospace!important;font-size:.9rem}
.gradio-container video{border-radius:16px}
.recent-card{align-items:center;padding:14px 18px!important;border:1px solid rgba(255,255,255,.09)!important;border-radius:18px!important;
 background:rgba(255,255,255,.035)!important;transition:transform .25s,border-color .25s,opacity .7s,box-shadow .25s}
.recent-card:hover{transform:translateY(-3px)!important;border-color:rgba(143,107,255,.3)!important;box-shadow:0 14px 34px rgba(0,0,0,.4)}
.cp-ripple{position:absolute;border-radius:50%;pointer-events:none;background:rgba(255,255,255,.35);transform:scale(0);animation:cp-rip .6s ease-out forwards}

/* ---------- result area spacing ---------- */
.gradio-container .cp-info{margin-top:8px!important;border-radius:14px!important;overflow:hidden}
.cp-info .prose{padding:18px 6px 8px!important;line-height:1.6}
.cp-info .prose:empty{padding:0!important}
.cp-info .prose p{margin:0 0 6px!important}
.cp-info .prose a{color:#ffb224;text-decoration:none;word-break:break-all}
.cp-info .prose a:hover{text-decoration:underline}
.gradio-container .generating,.gradio-container .pending{border-radius:16px!important;border-color:rgba(143,107,255,.45)!important}
.gradio-container .cp-status{margin-bottom:6px!important}

@keyframes cp-rip{to{transform:scale(1);opacity:0}}
@keyframes cp-pulse{50%{opacity:.35;transform:scale(.8)}}
@keyframes cp-shift{to{background-position:100% 0}}
@keyframes cp-scroll{to{transform:translateX(-50%)}}
@keyframes cp-play{from{left:0}to{left:100%}}
@keyframes cp-lit1{0%,11%{opacity:.3}13%,35%{opacity:1}38%,100%{opacity:.3}}
@keyframes cp-lit2{0%,45%{opacity:.3}47%,59%{opacity:1}62%,100%{opacity:.3}}
@keyframes cp-lit3{0%,71%{opacity:.3}73%,87%{opacity:1}90%,100%{opacity:.3}}
/* ---------- touch screens: no sticky hover states ---------- */
@media(hover:none){
 .card-radio label:hover,.chip label:hover,.cp-go:hover,.recent-card:hover,.cp-cta:hover,.cp-ghost:hover,#cp-fab:hover{transform:none!important}
 .card-radio label:hover{border-color:rgba(255,255,255,.09)!important}
 .recent-card:hover{border-color:rgba(255,255,255,.09)!important;box-shadow:none}
 .cp-go:hover{box-shadow:0 10px 40px rgba(143,107,255,.25)}
 .cp-go:hover::after{transform:translateX(-120%)}
 .cp-ghost:hover{background:rgba(255,255,255,.05)!important;color:inherit!important}
 .step:hover{border-color:rgba(255,255,255,.09)!important}
 .step:hover::before{opacity:0}
 #cp-nav button:not(.active):hover{background:transparent;color:#9494a8}
}
/* ---------- phones ---------- */
@media(max-width:640px){
 .gradio-container.gradio-container{padding:0 12px 96px!important}
 #cp-nav{padding:5px;gap:2px}
 #cp-nav .brand{display:none}
 #cp-nav button{padding:10px 10px;min-height:40px;font-size:12.5px}
 .stack-m{flex-direction:column!important}
 .stack-m>*{width:100%!important;flex:1 1 auto!important;min-width:0!important}
 .cp-hero{padding:92px 0 18px}
 .gradio-container .cp-hero p{font-size:1rem;margin-bottom:22px}
 .cp-cta{padding:14px 24px}
 .cp-tl{margin-top:32px}
 .cp-tl-track{height:44px}
 .cp-seg{top:6px;bottom:6px}
 .cp-seg b{display:none}
 .cp-marquee{margin-top:24px}
 .cp-track span{font-size:.9rem}
 .gradio-container .step{padding:18px 16px!important;margin-top:18px!important;border-radius:20px!important}
 .step-head{gap:12px}
 .step-no{font-size:2.1rem}
 .gradio-container .step-head h2{font-size:1.25rem}
 .card-radio label{min-height:64px;padding:14px 16px!important}
 .gradio-container input[type=text],.gradio-container input[type=number],.gradio-container textarea{font-size:16px!important}
 .gradio-container button.cp-go{min-height:58px}
 .cp-info .prose{padding:14px 2px 6px!important}
 .recent-card{flex-wrap:wrap!important;gap:10px!important}
 .recent-card button{width:100%!important;flex:1 1 100%!important}
 .recent-card .prose{word-break:break-word}
}
@media(max-width:380px){
 .gradio-container .cp-hero h1{font-size:2.3rem}
 .cp-pill{font-size:12px;padding:6px 12px}
 #cp-nav button{padding:10px 8px;font-size:12px}
}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}html{scroll-behavior:auto}
 body.cp-js .step,body.cp-js .reveal,body.cp-js .recent-card{opacity:1;transform:none}}
'''

CP_JS = r'''() => {
  document.body.classList.add('dark');
  const setIcon = () => {
    document.querySelectorAll('link[rel~="icon"], link[rel="shortcut icon"], link[rel="apple-touch-icon"]').forEach(l => l.remove());
    const l = document.createElement('link');
    l.rel = 'icon'; l.type = 'image/png'; l.href = 'data:image/png;base64,__FAVICON__';
    document.head.appendChild(l);
  };
  setIcon(); setTimeout(setIcon, 1500);
  let vp = document.querySelector('meta[name=viewport]');
  if (!vp) { vp = document.createElement('meta'); vp.name = 'viewport'; document.head.appendChild(vp); }
  vp.content = 'width=device-width, initial-scale=1, viewport-fit=cover';
  const boot = (tries) => {
    if (!document.getElementById('step-source') || !document.getElementById('cp-nav')) {
      if (tries < 40) setTimeout(() => boot(tries + 1), 250);
      return;
    }
    try {
      ['cp-nav', 'cp-progress', 'cp-fab'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.parentNode !== document.body) document.body.appendChild(el);
      });
      const jump = (id) => {
        if (id === 'top') { window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
        const t = document.getElementById(id);
        if (t) window.scrollTo({ top: t.getBoundingClientRect().top + window.scrollY - 86, behavior: 'smooth' });
      };
      const bar = document.getElementById('cp-progress');
      const fab = document.getElementById('cp-fab');
      const hero = document.querySelector('.cp-hero-inner');
      const onScroll = () => {
        const max = document.documentElement.scrollHeight - window.innerHeight;
        if (bar) bar.style.transform = 'scaleX(' + (max > 0 ? window.scrollY / max : 0) + ')';
        document.body.classList.toggle('cp-scrolled', window.scrollY > 24);
        if (fab) fab.classList.toggle('show', window.scrollY > 500);
        if (hero) {
          hero.style.transform = 'translateY(' + Math.min(window.scrollY, 600) * 0.18 + 'px)';
          hero.style.opacity = Math.max(0, 1 - window.scrollY / 650);
        }
      };
      window.addEventListener('scroll', onScroll, { passive: true });
      onScroll();

      document.addEventListener('click', (e) => {
        const j = e.target.closest('[data-jump]');
        if (j) { e.preventDefault(); jump(j.dataset.jump); }
        if (e.target.closest('#make-btn')) setTimeout(() => jump('step-create'), 150);
      });
      document.addEventListener('keydown', (e) => {
        const j = e.target.closest && e.target.closest('[role=button][data-jump]');
        if (j && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); jump(j.dataset.jump); }
      });
      document.addEventListener('pointerdown', (e) => {
        const host = e.target.closest('button, .card-radio label, .chip label');
        if (!host) return;
        const r = host.getBoundingClientRect(), d = Math.max(r.width, r.height) * 2;
        const s = document.createElement('span');
        s.className = 'cp-ripple';
        s.style.cssText = 'width:' + d + 'px;height:' + d + 'px;left:' + (e.clientX - r.left - d / 2) + 'px;top:' + (e.clientY - r.top - d / 2) + 'px';
        host.appendChild(s);
        setTimeout(() => s.remove(), 650);
      });
      document.addEventListener('pointermove', (e) => {
        const st = e.target.closest && e.target.closest('.step');
        if (!st) return;
        const r = st.getBoundingClientRect();
        st.style.setProperty('--mx', (e.clientX - r.left) + 'px');
        st.style.setProperty('--my', (e.clientY - r.top) + 'px');
      });

      document.body.classList.add('cp-js');
      const io = new IntersectionObserver((es) => es.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      }), { threshold: 0.1 });
      const seen = new WeakSet();
      const watch = () => document.querySelectorAll('.step, .reveal, .recent-card').forEach(el => {
        if (!seen.has(el)) { seen.add(el); io.observe(el); }
      });
      watch();
      new MutationObserver(watch).observe(document.body, { childList: true, subtree: true });

      const links = document.querySelectorAll('#cp-nav [data-jump]');
      const spy = new IntersectionObserver((es) => es.forEach(en => {
        if (en.isIntersecting) {
          links.forEach(l => l.classList.toggle('active', l.dataset.jump === en.target.id));
          const a = document.querySelector('#cp-nav button.active'), box = a && a.parentNode;
          if (a && box) box.scrollTo({ left: a.offsetLeft - (box.clientWidth - a.offsetWidth) / 2, behavior: 'smooth' });
        }
      }), { rootMargin: '-45% 0px -50% 0px' });
      links.forEach(l => { const t = document.getElementById(l.dataset.jump); if (t) spy.observe(t); });
    } catch (err) {
      document.body.classList.remove('cp-js');
      console.error(err);
    }
  };
  setTimeout(() => boot(0), 300);
}
'''

FAVICON_B64 = "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"

CP_CHROME = """
<div id="cp-progress"></div>
<div id="cp-nav">
  <span class="brand" data-jump="top" role="button" tabindex="0" title="Back to top"><i>&#9670;</i>WannaClip</span>
  <div class="links">
    <button data-jump="step-source">Source</button>
    <button data-jump="step-format">Format</button>
    <button data-jump="step-moments">Moments</button>
    <button data-jump="step-polish">Polish</button>
    <button data-jump="step-create">Create</button>
    <button data-jump="step-recent">Recent</button>
  </div>
</div>
<button id="cp-fab" data-jump="top" aria-label="Back to top">&uarr;</button>
"""

CP_HERO = """
<div class="cp-hero"><div class="cp-hero-inner">
  <span class="cp-pill"><b></b>Ready when you are</span>
  <h1>Clip the moment.<br><span class="cp-grad">Make it go viral.</span></h1>
  <p>Paste a link or type a topic. Get a captioned clip back.</p>
  <button class="cp-cta" data-jump="step-source">Start clipping <span>&darr;</span></button>
  <div class="cp-tl">
    <div class="cp-tl-track">
      <i class="cp-seg s1"><b>12:12&ndash;15:09</b></i>
      <i class="cp-seg s2"><b>16:00&ndash;16:54</b></i>
      <i class="cp-seg s3"><b>20:30</b></i>
      <span class="cp-head"></span>
    </div>
    <div class="cp-tl-cap"><span>0:00</span><span>full video</span><span>end</span></div>
  </div>
</div></div>
"""

_words = ["Shorts", "Highlights", "Goals", "Reactions", "Podcasts", "Reels", "TikTok", "Moments", "Captions"]
CP_MARQUEE = ('<div class="cp-marquee"><div class="cp-track">'
              + "".join(f"<span>{w}</span>" for w in _words * 2)
              + "</div></div>")


def step_head(n, title, sub=""):
    p = f"<p>{sub}</p>" if sub else ""
    return gr.HTML(f'<div class="step-head"><span class="step-no">{n:02d}</span><div><h2>{title}</h2>{p}</div></div>')


CP_JS = CP_JS.replace("__FAVICON__", FAVICON_B64)

with gr.Blocks(title="Wanna Clip") as demo:
    gr.HTML("<style>" + CP_CSS + "</style>", elem_classes=["cp-chrome"])
    gr.HTML(CP_CHROME, elem_classes=["cp-chrome"])
    gr.HTML(CP_HERO)
    gr.HTML(CP_MARQUEE, elem_classes=["reveal"])

    with gr.Column(elem_classes=["step"], elem_id="step-source"):
        step_head(1, "Pick your video", "Paste a link, or type a topic and we'll find one.")
        with gr.Row(elem_classes=["stack-m"]):
            q = gr.Textbox(label="Topic or title", placeholder="Type a topic or video title", scale=3)
            link = gr.Textbox(label="YouTube link", placeholder="Paste a YouTube link", scale=3)
        with gr.Accordion("Search options", open=False):
            min_views = gr.Number(label="Minimum views", value=100000, precision=0)
            own_channel = gr.Textbox(label="Channel name", placeholder="Separate several names with commas")
            cc = gr.Checkbox(label="Creative Commons only", value=False)

    with gr.Column(elem_classes=["step"], elem_id="step-format"):
        step_head(2, "Choose a format", "Tap one.")
        video_format = gr.Radio(FORMAT_CHOICES, value=FORMAT_CHOICES[0], show_label=False,
                                elem_classes=["card-radio", "fmt-cards"])

    with gr.Column(elem_classes=["step"], elem_id="step-moments"):
        step_head(3, "Pick the moments", "Let the app choose, or give exact timestamps.")
        clip_type = gr.Radio(CLIP_TYPES, value=CLIP_TYPES[0], show_label=False,
                             elem_classes=["card-radio", "type-cards"])
        mode = gr.Dropdown(MODE_CHOICES, value=MODE_CHOICES[0], label="What to cut", visible=False)
        clip_type.change(lambda c: gr.update(visible=str(c).startswith("Multiple")), clip_type, mode)
        stamps = gr.Textbox(label="Exact timestamps (optional)",
                            placeholder="12:12-15:09, 16:00-16:54   or just   12:12",
                            info="A single time starts there and runs for the length below.")
        length = gr.Slider(30, 600, value=90, step=5, label="Length (seconds)")

    with gr.Column(elem_classes=["step"], elem_id="step-polish"):
        step_head(4, "Add the polish", "Tap to switch on or off.")
        with gr.Row():
            fx_glitch = gr.Checkbox(label="Glitch transitions", value=True, elem_classes=["chip"])
            fx_pop = gr.Checkbox(label="Captions that pop", value=True, elem_classes=["chip"])
        with gr.Accordion("Transition sound", open=False):
            sound = gr.File(label="Sound file (optional)",
                            type="filepath", file_types=[".mp3", ".wav", ".m4a", ".ogg"])
    fx_comps = [fx_glitch, fx_pop]

    with gr.Column(elem_classes=["step"], elem_id="step-create"):
        step_head(5, "Make it")
        btn = gr.Button("Make clip", variant="primary", elem_id="make-btn", elem_classes=["cp-go"])
        status = gr.Textbox(label="Status", interactive=False, elem_classes=["cp-status"])
        info_md = gr.Markdown(elem_classes=["cp-info"])
        with gr.Row(elem_classes=["stack-m"]):
            video = gr.Video(label="Result")
            file = gr.File(label="Download")

    gr.HTML('<div class="step-head"><span class="step-no">06</span><div><h2>Recent</h2>'
            '<p>Your last videos. Tap to run again with new moments.</p></div></div>',
            elem_id="step-recent", elem_classes=["reveal"])
    # ---- Last used (3 most recent videos) with a "Use again" button each ----
    recent_state = gr.State([])
    recent_rows, recent_mds, recent_btns = [], [], []
    for _ in range(3):
        with gr.Row(visible=False, elem_classes=["recent-card"]) as r:
            m = gr.Markdown()
            b = gr.Button("Use again", scale=0, min_width=150, elem_classes=["cp-ghost"])
        recent_rows.append(r)
        recent_mds.append(m)
        recent_btns.append(b)
    recent_outs = [recent_state] + [c for pair in zip(recent_rows, recent_mds) for c in pair]

    def refresh_recent():
        items = recent_videos(3)
        out = [items]
        for i in range(3):
            if i < len(items):
                it = items[i]
                moms = it.get("moments") or []
                when = (it.get("last_used") or "").replace("T", " ")[:16]
                seen = ", ".join(fmt_time(m) for m in moms[-8:]) or "none"
                more = " …" if len(moms) > 8 else ""
                out += [gr.Row(visible=True),
                        gr.Markdown(f"**{it.get('title', '')}**  \n"
                                    f"{len(moms)} moment{'s' if len(moms) != 1 else ''} used "
                                    f"({seen}{more}) · last used {when}  \n"
                                    f"{it.get('url', '')}")]
            else:
                out += [gr.Row(visible=False), gr.Markdown("")]
        return out

    def make_reuse(i):
        def f(items, length, cc, min_views, own_channel, video_format, clip_type, mode, sound, *fx):
            if i >= len(items):
                yield "Nothing to reuse yet.", None, None, ""
                return
            yield from run("", items[i]["url"], length, cc, min_views, own_channel, video_format, clip_type, mode, sound, "", *fx)
        return f

    ins = [q, link, length, cc, min_views, own_channel, video_format, clip_type, mode, sound, stamps] + fx_comps
    outs = [status, video, file, info_md]
    btn.click(run, ins, outs).then(refresh_recent, None, recent_outs)
    q.submit(run, ins, outs).then(refresh_recent, None, recent_outs)
    link.submit(run, ins, outs).then(refresh_recent, None, recent_outs)
    for i, b in enumerate(recent_btns):
        b.click(make_reuse(i), [recent_state, length, cc, min_views, own_channel, video_format, clip_type, mode, sound] + fx_comps, outs
                ).then(refresh_recent, None, recent_outs)
    demo.load(refresh_recent, None, recent_outs)
    demo.load(None, None, None, js=CP_JS)

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _fav = next((p for p in (os.path.join(_here, "favicon.png"), os.path.join(_here, "assets", "favicon.png"))
                 if os.path.exists(p)), None)
    demo.launch(inbrowser=True, favicon_path=_fav)     # tab icon (your own logo)
