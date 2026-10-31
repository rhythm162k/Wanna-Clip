# Contributing to Wanna Clip

Thanks for wanting to help! Wanna Clip is a small, single-file project, so contributing is simple. Bug reports, ideas, docs fixes and code are all welcome.

By taking part, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- **Report a bug.** Use the [bug report form](../../issues/new/choose) and include the details listed below.
- **Suggest a feature.** Open a feature request and explain the problem you want solved, not just the solution.
- **Improve the docs.** Typos, unclear steps and missing examples are all fair game.
- **Fix something.** Look for issues labeled `good first issue` or `help wanted`.
- **Test on your system.** Reports from Windows, Linux and different Python versions are very valuable.

## Reporting a bug

Please include:

1. Your operating system and Python version (`python --version`)
2. Your `yt-dlp` version (`pip show yt-dlp`)
3. What you did, what you expected, and what happened instead
4. The full error text from the terminal
5. The video link or topic you used, if it's relevant

> Before reporting a download error (HTTP 403, sign-in errors), run `pip install -U yt-dlp` and try again. YouTube changes often and this fixes most of them.

## Setting up for development

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/Wanna-Clip.git
cd Wanna-Clip

# 2. Create an isolated environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app opens at http://127.0.0.1:7860. FFmpeg is bundled through `imageio-ffmpeg`, so you don't need to install it.

## Making a change

1. Create a branch from `main`: `git checkout -b my-change`
2. Make your change in small, focused commits.
3. Run the app and try your change by hand, with both **Short** and **Regular** formats if it touches rendering.
4. Push your branch and open a pull request.

### Guidelines

- **Keep it focused.** One fix or feature per pull request.
- **Match the existing style.** The project lives in `app.py`; follow the naming and layout already there.
- **No secrets or large files.** Never commit API keys, `.env` files, videos, model files or a `venv/` folder.
- **Keep it local.** The project promises no API keys and no data leaving the user's computer. Changes that break this promise need a discussion first.
- **Update the docs.** If you add or change a setting, update the README too.
- **Be respectful of copyright.** Don't add features whose main purpose is to repost other people's content.

### Pull request checklist

- [ ] I ran `python app.py` and tested the change
- [ ] I didn't commit `venv/`, `.env`, videos or model files
- [ ] I updated the README if behavior or settings changed
- [ ] My pull request describes what changed and why

## Ideas that would be great to have

- More caption styles and fonts
- A language selector for Whisper
- Support for video sources other than YouTube
- A Dockerfile
- Optional GPU acceleration
- Automated tests for timestamp parsing and history handling

Not sure if your idea fits? Open an issue and ask first.

## Questions

Open a [Discussion](../../discussions) if they're enabled, or an issue labeled `question`.
