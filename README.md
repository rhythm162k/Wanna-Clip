<div align="center">

<img src="assets/banner.svg" alt="Viral Clip Agent: find the moment, add the captions, ship the clip" width="100%">

<br>

<a href="#-quick-start"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-8f6bff?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://www.gradio.app"><img alt="Gradio" src="https://img.shields.io/badge/UI-Gradio-b266ff?style=for-the-badge"></a>
<a href="https://github.com/yt-dlp/yt-dlp"><img alt="yt-dlp" src="https://img.shields.io/badge/downloads-yt--dlp-7c5cff?style=for-the-badge"></a>
<a href="https://github.com/SYSTRAN/faster-whisper"><img alt="faster-whisper" src="https://img.shields.io/badge/captions-faster--whisper-9d7dff?style=for-the-badge"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-ffb224?style=for-the-badge"></a>

<br><br>

**Turn any YouTube video into a captioned, share-ready short.**<br>
Paste a link or type a topic. Viral Clip Agent finds the best moments, cuts them, adds word-by-word captions, and hands you an MP4.

[Features](#-features) · [How it works](#-how-it-works) · [Quick start](#-quick-start) · [Usage](#-usage) · [Configuration](#-configuration) · [Troubleshooting](#-troubleshooting)

</div>

<img src="assets/divider.svg" width="100%" alt="">

<!--
  Add a screenshot of the app here:
  1. Save it as assets/screenshot.png
  2. Delete the comment markers around the block below

<div align="center">
  <img src="assets/screenshot.png" alt="Viral Clip Agent interface" width="860">
</div>
-->

## ✨ Features

<table>
  <tr>
    <td width="33%" valign="top">
      <h4>🔎 Finds the video for you</h4>
      Type a topic and it searches for a popular video from the last two years. It skips music and copyrighted uploads, and other people's reaction videos.
    </td>
    <td width="33%" valign="top">
      <h4>🎯 Picks the best moments</h4>
      Uses YouTube's <b>Most replayed</b> data when available, and falls back to the loudest, most exciting parts of the audio.
    </td>
    <td width="33%" valign="top">
      <h4>⏱️ Exact timestamps</h4>
      Know the moment you want? Enter <code>12:12-15:09, 16:00-16:54</code> and it cuts and joins those ranges, skipping auto-detection.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>💬 Captions that pop</h4>
      Word-level captions generated locally with Whisper, animated as each word is spoken.
    </td>
    <td valign="top">
      <h4>⚡ Glitch transitions</h4>
      Multi-moment clips are stitched with glitch cuts. Add your own transition sound if you like.
    </td>
    <td valign="top">
      <h4>⚽ Football goal mode</h4>
      Detects goals from commentary, crowd roars and referee whistles, and tells them apart from fouls.
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h4>📱 Short or widescreen</h4>
      Export vertical <b>1080×1920</b> (up to 3 min) or horizontal <b>1920×1080</b> (up to 10 min).
    </td>
    <td valign="top">
      <h4>🔁 Never repeats itself</h4>
      Remembers which moments of a video it already used, so running it again gives you fresh ones.
    </td>
    <td valign="top">
      <h4>🖥️ Runs on your machine</h4>
      CPU-only, no GPU, no API keys, and no separate FFmpeg install. Your videos never leave your computer.
    </td>
  </tr>
</table>

<img src="assets/divider.svg" width="100%" alt="">

## 🧭 How it works

<img src="assets/how-it-works.svg" alt="Source, Analyze, Pick, Polish, Export" width="100%">

1. **Source:** you give it a YouTube link, or a topic to search for.
2. **Analyze:** it listens to the whole video with a fast Whisper model, looks for audio peaks, and reads YouTube's *Most replayed* data when the video has it.
3. **Pick:** it chooses the strongest moments, or uses the timestamps you typed.
4. **Polish:** it renders the clip with word-by-word captions, optional glitch transitions and your own sound.
5. **Export:** you get an MP4 in the interface, and a copy is saved to your `Downloads` folder.

<img src="assets/divider.svg" width="100%" alt="">

## 🚀 Quick start

**Requirements:** Python 3.10 or newer, and about 1 GB of free disk space for the speech models, which download on first run.

```bash
# 1. Get the code
git clone https://github.com/YOUR-USERNAME/viral-clip-agent.git
cd viral-clip-agent

# 2. Create an isolated environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
python app.py
```

Your browser opens at **http://127.0.0.1:7860**. FFmpeg is bundled through `imageio-ffmpeg`, so there is nothing else to install.

> [!NOTE]
> The first clip is slower than the rest because the Whisper models (about 460 MB for captions, plus a small one for analysis) are downloaded once and then cached.

> [!TIP]
> YouTube changes often. If downloads start failing, update the downloader: `pip install -U yt-dlp`.

<img src="assets/divider.svg" width="100%" alt="">

## 🎬 Usage

1. **Pick your video.** Paste a YouTube link, or type a topic.
2. **Choose a format.** Short (vertical) or Regular video (horizontal).
3. **Pick the moments.** Make one clip, or join multiple moments. Optionally enter exact timestamps.
4. **Add the polish.** Switch glitch transitions and pop captions on or off, and optionally add a transition sound.
5. **Make clip.** Watch progress in the status box. The result appears below and is saved to `~/Downloads`.

### Clip types

| Option | What you get |
| --- | --- |
| **One clip** | The single best moment, using *Most replayed* data when available, otherwise the loudest part of the audio. |
| **Multiple moments joined** | Several strong moments joined with transitions. Choose *Auto-detect*, *Football goals (one match)* or *Best moments (one topic)* under **What to cut**. |

### Exact timestamps (optional)

Leave the box empty and the app chooses for you. Fill it in and the app uses exactly what you typed.

| You type | What happens |
| --- | --- |
| `12:12` | One clip starting at 12:12 and running for the **Length** you set |
| `12:12-15:09` | One clip from 12:12 to 15:09 |
| `12:12-15:09, 16:00-16:54` | Both ranges cut and joined, in the order you typed them |
| `12:12-15:09, 20:30` | Mix ranges and single points freely |

Hours work too (`1:02:03`), and so does `12:12 to 15:09`. The total must fit the format limit (3 minutes for Short, 10 for Regular). If it doesn't, the app tells you instead of silently trimming your ranges.

### Search options

Under **Search options** you can set a minimum view count (default 100,000), restrict results to Creative Commons videos, and name a channel whose own reaction content should be allowed.

### Where files go

| What | Where |
| --- | --- |
| Finished clips | Shown in the app and copied to `~/Downloads` |
| Usage history (for "Recent") | `~/ViralClips/used_videos_v4.json` |
| Temporary working files | Your system temp folder, cleaned up automatically |

<img src="assets/divider.svg" width="100%" alt="">

## ⚙️ Configuration

Everything lives in `app.py`. These are the settings people most often change:

| Setting | Default | What it does |
| --- | --- | --- |
| `get_whisper()` model | `"small"` | Caption accuracy. Try `"base"` for speed or `"medium"` for accuracy. |
| `ANALYSIS_MODEL` | `"tiny"` | The fast model used to find moments. `"base"` is slower but more accurate. |
| `MAX_AGE_DAYS` | `730` | How far back topic search looks for videos. |
| `FORMATS` | Short / Regular | Output size, caption size and position, and maximum length. |
| `FONT_PATHS` | macOS, Windows, Linux | Fonts tried for captions, in order. |
| `OUT_DIR` | `~/ViralClips` | Where history is stored. |

Note that videos are downloaded at up to 720p to keep things fast, then rendered to the output size.

<img src="assets/divider.svg" width="100%" alt="">

## 🛠️ Troubleshooting

<details>
<summary><b>Downloads fail, or I get HTTP 403 / sign-in errors</b></summary>

Update the downloader, since YouTube changes frequently:

```bash
pip install -U yt-dlp
```
</details>

<details>
<summary><b>The first run is very slow or looks stuck</b></summary>

The speech models download once on first use. After that they are cached and runs are much faster. Long videos take longer because the whole video is analyzed.
</details>

<details>
<summary><b>Captions look tiny or use the wrong font</b></summary>

The app looks for bold system fonts in `FONT_PATHS`. If none exist on your system, it falls back to a basic built-in font. Add the path to any bold `.ttf` file to that list.
</details>

<details>
<summary><b>I want a video's moments to be picked fresh again</b></summary>

The app remembers moments it has used so repeats give new ones. To reset everything, delete `~/ViralClips/used_videos_v4.json`.
</details>

<details>
<summary><b>The page looks unstyled or animations are missing</b></summary>

The interface styling uses features of recent Gradio versions. Update with `pip install -U gradio`, then hard-refresh the page.
</details>

<img src="assets/divider.svg" width="100%" alt="">

## 📁 Project structure

```
viral-clip-agent/
├── app.py              # the whole app: search, analysis, rendering, interface
├── requirements.txt    # Python dependencies
├── assets/             # README graphics
├── LICENSE
└── README.md
```

## 🧱 Built with

[Gradio](https://www.gradio.app) for the interface · [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription · [FFmpeg](https://ffmpeg.org) via [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) for video · [Pillow](https://python-pillow.org) and [NumPy](https://numpy.org) for captions and effects.

## ⚖️ Responsible use

This tool makes it easy to cut other people's videos. Only clip content you own, have permission to use, or that is licensed for reuse (the **Creative Commons only** option helps here). You are responsible for following YouTube's Terms of Service and the copyright rules where you live.

## 📄 License

Released under the [MIT License](LICENSE).

<br>

<div align="center">
  <img src="assets/footer.svg" alt="Viral Clip Agent" width="100%">
  <br><br>
  Built by <b>Md. Sami-El Bashar Rhythm</b><br>
  <sub>If this saved you time, a ⭐ on the repo is much appreciated.</sub>
</div>
