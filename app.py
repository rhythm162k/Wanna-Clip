"""
Viral Clip Agent  -  runs fully locally on a Mac (Apple Silicon / M1).

Type a topic  ->  finds the most-viewed video of the last 12 months about ONLY that topic
-> finds the most "replayed" moment -> cuts a 60-90s vertical clip -> burns in
viral-style captions (big uppercase, 3 words at a time, active word highlighted)
-> gives you an MP4 to download.

No API keys. No external software to install (ffmpeg is bundled via pip).
"""
import os
import re
import json
import random
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
HISTORY_FILE = os.path.join(OUT_DIR, "used_videos.json")
MAX_AGE_DAYS = 730  # 2 years
MAX_DURATION_ANY = 3600            # 1 h  (full download needed)
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

    data["_last_reset"] = last.isoformat()
    # migrate old entries (video id -> title) to the richer format
    for k, v in list(data.items()):
        if not k.startswith("_") and not isinstance(v, dict):
            data[k] = {"title": str(v), "url": f"https://www.youtube.com/watch?v={k}",
                       "channel": "", "moments": [], "last_used": None, "legacy": True}
    save_history(data)
    return data


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
        return len(k) >= 6 and k in joined               # "ishowspeed" == "I Show Speed"

    score = sum(1 for k in kws if hit(k)) / len(kws)
    if norm(query) in norm(text):          # whole sentence appears -> perfect
        score = 1.0
    return score


# Titles/tags that signal "someone reacting to / reviewing / commenting on another video"
REACTION_RE = re.compile(
    r"\b(react\w*|reaction\w*|respond\w*|response to|reviewing|my thoughts|"
    r"watching|watch party|watch along|watchalong)\b", re.I)


def subject_terms(query, own_channel=""):
    """Name(s) that identify the subject's OWN channel (e.g. 'ishowspeed')."""
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
def _find_downloaded(workdir):
    ok = (".mp4", ".mkv", ".webm", ".mov")
    files = [f for f in os.listdir(workdir)
             if f.startswith("source") and f.lower().endswith(ok)]
    files.sort(key=lambda f: os.path.getsize(os.path.join(workdir, f)), reverse=True)
    return os.path.join(workdir, files[0]) if files else None


def _download_try(info, workdir, section):
    base = {
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(workdir, "source.%(ext)s"),
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
        for f in os.listdir(workdir):           # clean partial files
            if f.startswith("source"):
                try:
                    os.remove(os.path.join(workdir, f))
                except OSError:
                    pass
        try:
            with yt_dlp.YoutubeDL({**base, **extra}) as y:
                y.download([info["webpage_url"]])
            path = _find_downloaded(workdir)
            if path:
                return path
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All download methods failed (YouTube blocked it). Last error: {last_err}")


def download(info, workdir, section=None):
    """Returns (file, offset). With a section, the file starts at `offset` seconds of the video."""
    if section:
        try:
            return _download_try(info, workdir, section), section[0]
        except Exception:
            pass                                 # fall back to the full video below
    return _download_try(info, workdir, None), 0


# ----------------------------------------------------------------------------
# 3) FIND THE MOST VIRAL MOMENT
#    Primary: YouTube's "Most replayed" heatmap (real audience data).
#    Fallback: loudest / most energetic stretch of audio.
# ----------------------------------------------------------------------------
MAX_MOMENTS = 5      # at most this many different moments per video
MIN_RATIO = 0.4      # a moment must have >= 40% of the best moment's score


def rank_windows(scores, length, max_n=MAX_MOMENTS, min_ratio=MIN_RATIO):
    """All distinct (non-overlapping) viral moments, best first: [(start_second, score), ...]"""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return [(0, 0.0)]
    length = max(1, min(length, n))
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


def energy_ranking(src, length):
    """Fallback (no heatmap): ranked moments by loudest / most energetic audio."""
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    secs = len(audio) // 8000
    rms = np.sqrt((audio[:secs * 8000].reshape(secs, 8000) ** 2).mean(axis=1))
    return rank_windows(rms, length)


def choose_moment(entry, ranked, length):
    """Best moment of this video that hasn't been used yet.
    Returns (start, number_of_this_moment, total_moments, used_list)."""
    used = list(entry.get("moments") or [])
    if entry.get("legacy") and not used and ranked:
        used.append(ranked[0][0])          # old versions always used the top moment
    for rank, (start, _score) in enumerate(ranked, 1):
        if all(abs(start - u) >= length for u in used):
            return start, rank, len(ranked), used
    raise RuntimeError(
        f"You've already made clips from all {len(ranked)} viral moment(s) of this video. "
        "Pick another video.")


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
]
STRIP_H, STRIP_Y = 280, 1330
YELLOW, WHITE = (255, 222, 0, 255), (255, 255, 255, 255)


def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


_strip_cache = {}


def caption_strip(texts, active):
    key = (tuple(texts), active)
    if key in _strip_cache:
        return _strip_cache[key]
    texts = [t.upper() for t in texts]
    size = 120
    while True:
        font = load_font(size)
        space = font.getlength(" ")
        widths = [font.getlength(t) for t in texts]
        total = sum(widths) + space * (len(texts) - 1)
        if total <= W - 100 or size <= 50:
            break
        size -= 6
    img = Image.new("RGBA", (W, STRIP_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = (W - total) / 2
    for i, t in enumerate(texts):
        d.text((x, STRIP_H // 2), t, font=font, anchor="lm",
               fill=YELLOW if i == active else WHITE,
               stroke_width=max(6, size // 11), stroke_fill=(0, 0, 0, 255))
        x += widths[i] + space
    arr = np.array(img)
    res = (arr[..., :3].astype(np.uint16), arr[..., 3:4].astype(np.uint16))
    _strip_cache[key] = res
    return res


# ----------------------------------------------------------------------------
# 6) RENDER: 9:16, blurred background, video centered, captions burned in
# ----------------------------------------------------------------------------
def render(src, start, dur, groups, out_path):
    vf = (f"split[a][b];"
          f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bg];"
          f"[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={FPS}")
    dec = subprocess.Popen(
        [FFMPEG, "-v", "error", "-ss", str(start), "-t", str(dur), "-i", src,
         "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-ss", str(start), "-t", str(dur), "-i", src,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    fsize = W * H * 3
    gi, i = 0, 0
    while True:
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = i / FPS
        if groups:
            while gi + 1 < len(groups) and groups[gi + 1]["start"] <= t:
                gi += 1
            g = groups[gi]
            if g["start"] <= t < g["end"]:
                active = max(k for k, w in enumerate(g["words"]) if w[1] <= t)
                rgb, alpha = caption_strip([w[0] for w in g["words"]], active)
                reg = frame[STRIP_Y:STRIP_Y + STRIP_H].astype(np.uint16)
                frame[STRIP_Y:STRIP_Y + STRIP_H] = ((reg * (255 - alpha) + rgb * alpha) // 255).astype(np.uint8)
        enc.stdin.write(frame.tobytes())
        i += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()


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


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def run(query, link, length, cc_only, min_views, own_channel):
    query = (query or "").strip()
    link = (link or "").strip()
    if not query and not link:
        yield "Type a topic/title OR paste a YouTube link (at least one is needed).", None, None, ""
        return
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
                f"{info['webpage_url']}")

        length = int(length)
        total = int(info["duration"])
        entry = get_entry(info)                  # which moments of this video were used before

        ranked = heatmap_ranking(info, length)
        if ranked is not None:
            how = "YouTube 'Most replayed' heatmap"
            start, nth, count, used = choose_moment(entry, ranked, length)
            dur = min(length, total - start)
            yield "⬇️ Downloading just the best part of the video…", None, None, meta
            src, offset = download(info, workdir, (max(0, start - 3), min(total, start + dur + 3)))
        else:
            how = "loudest / most energetic audio"
            yield "⬇️ Downloading the video…", None, None, meta
            src, offset = download(info, workdir)
            yield "🔥 Finding the most viral moments…", None, None, meta
            ranked = energy_ranking(src, length)
            start, nth, count, used = choose_moment(entry, ranked, length)
            dur = min(length, total - start)
        clip_start = start - offset            # position inside the downloaded file
        meta += (f"  \nMoment {nth} of {count} found · starts at {start // 60}:{start % 60:02d} "
                 f"· via {how}")

        yield "🎙️ Transcribing speech (first run downloads the model)…", None, None, meta
        words = transcribe(src, clip_start, dur, workdir)
        groups = make_groups(words)

        yield "🎬 Rendering vertical clip with viral captions (a few minutes)…", None, None, meta
        outdir = tempfile.mkdtemp(prefix="viralclip_out_")      # temporary, not your folders
        out = unique_path(outdir, clean_filename(info.get("title"), info.get("id")))
        render(src, clip_start, dur, groups, out)

        record_use(info, used, start)            # now it shows up in "Last used"
        yield "✅ Done! Use the Download button to save your video.", out, out, meta
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

with gr.Blocks(title="Viral Clip Agent") as demo:
    gr.Markdown("# 🔥 Viral Clip Agent\n"
                "Paste a **YouTube link**, or type a **topic / title** → get a captioned 9:16 short.  \n"
                "Fill in at least one. If you fill in both, the link is used.  \n"
                "Using the same video again gives you its next most-viral moment.")
    with gr.Row():
        q = gr.Textbox(label="Topic or title (optional if you paste a link)",
                       placeholder="e.g. Spiderman fights Superman", scale=3)
        link = gr.Textbox(label="YouTube link (optional if you type a topic)",
                          placeholder="https://www.youtube.com/watch?v=...", scale=3)
    with gr.Row():
        length = gr.Slider(60, 90, value=75, step=5, label="Clip length (seconds)", scale=2)
        min_views = gr.Number(label="Minimum views (topic search only)", value=100000,
                              precision=0, scale=1)
    own_channel = gr.Textbox(label="Subject's own channel name (optional, topic search only)",
                             placeholder="e.g. Spiderman, Superman  - lets their own reaction videos through")
    cc = gr.Checkbox(label="Only Creative Commons videos (topic search only)", value=False)
    btn = gr.Button("Make my viral clip", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    info_md = gr.Markdown()
    with gr.Row():
        video = gr.Video(label="Result")
        file = gr.File(label="Download")

    # ---- Last used (3 most recent videos) with a "Use again" button each ----
    gr.Markdown("### 🕘 Last used")
    recent_state = gr.State([])
    recent_rows, recent_mds, recent_btns = [], [], []
    for _ in range(3):
        with gr.Row(visible=False) as r:
            m = gr.Markdown()
            b = gr.Button("🔁 Use again", scale=0, min_width=150)
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
                n = len(it.get("moments") or [])
                when = (it.get("last_used") or "").replace("T", " ")[:16]
                out += [gr.Row(visible=True),
                        gr.Markdown(f"**{it.get('title', '')}**  \n"
                                    f"{n} moment{'s' if n != 1 else ''} used · last used {when}")]
            else:
                out += [gr.Row(visible=False), gr.Markdown("")]
        return out

    def make_reuse(i):
        def f(items, length, cc, min_views, own_channel):
            if i >= len(items):
                yield "Nothing to reuse yet.", None, None, ""
                return
            yield from run("", items[i]["url"], length, cc, min_views, own_channel)
        return f

    ins = [q, link, length, cc, min_views, own_channel]
    outs = [status, video, file, info_md]
    btn.click(run, ins, outs).then(refresh_recent, None, recent_outs)
    q.submit(run, ins, outs).then(refresh_recent, None, recent_outs)
    link.submit(run, ins, outs).then(refresh_recent, None, recent_outs)
    for i, b in enumerate(recent_btns):
        b.click(make_reuse(i), [recent_state, length, cc, min_views, own_channel], outs
                ).then(refresh_recent, None, recent_outs)
    demo.load(refresh_recent, None, recent_outs)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
