from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import threading
import traceback
import uuid

import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context
from mutagen import File as MutagenFile
from mutagen.id3 import APIC
from werkzeug.utils import secure_filename

from alignment_cache import AlignmentCache
from alignment_engine import ALIGNMENT_ENGINE_VERSION, runtime_info
from lrc_formats import build_outputs, parse_lyrics
from lrc_maker import generate_elrc

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("lrc_studio.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
DOWNLOADS_DIR = BASE_DIR / "downloads"
STATE_DIR = BASE_DIR / ".lrc_extandator"
UPLOAD_FOLDER.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _api_token() -> str:
    token_file = STATE_DIR / "api-token.txt"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    token_file.write_text(token, encoding="utf-8")
    return token


API_TOKEN = _api_token()


def _is_v1_authorized() -> bool:
    supplied = request.headers.get("X-LRC-Token", "")
    return secrets.compare_digest(supplied, API_TOKEN)


@app.after_request
def _cors_local_api(response):
    # V1 is token-protected. Allow the future Better Lyrics extension to call
    # localhost without opening the service to arbitrary LAN clients.
    origin = request.headers.get("Origin", "")
    if request.path.startswith("/v1/") and (
        origin == "https://music.youtube.com"
        or origin.startswith("moz-extension://")
        or origin.startswith("chrome-extension://")
    ):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-LRC-Token"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


def _result_payload(result, source_name, cover=None, mp3=None):
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Не удалось синхронизировать текст")
    base_name = Path(source_name).stem
    payload = {
        "type": "result",
        "elrc": result["elrc"],
        "elrc_compatible": result.get("elrc_compatible", result["elrc"]),
        "lrc": result["lrc"],
        "filename": f"{base_name}.elrc",
        "lrc_filename": f"{base_name}.lrc",
        "cover": cover,
        "lines": result["lines"],
        "artist": result.get("artist", ""),
        "title": result.get("title", ""),
        "album": result.get("album", ""),
        "validation_issues": result.get("validation_issues", []),
        "quality": result.get("quality", {}),
        "alignment": result.get("alignment", {}),
        "engine_version": result.get("engine_version", ALIGNMENT_ENGINE_VERSION),
    }
    if mp3:
        payload["mp3"] = mp3
    return payload


def _parse_demucs(value, default="auto"):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"auto", "adaptive"}:
        return "auto"
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _stream_elrc(audio_path, source_name, *, cover=None, cleanup=False, mp3=None, **options):
    events: queue.Queue[object] = queue.Queue()
    finished = object()

    def on_progress(percent, message):
        events.put({"type": "progress", "percent": percent, "message": message})

    def worker():
        try:
            result = generate_elrc(audio_path, progress_callback=on_progress, **options)
            events.put(_result_payload(result, source_name, cover=cover, mp3=mp3))
        except Exception as exc:
            logger.error("Generation failed: %s", traceback.format_exc())
            events.put({"type": "error", "message": str(exc)})
        finally:
            if cleanup:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            events.put(finished)

    threading.Thread(target=worker, name="elrc-generator", daemon=True).start()
    while True:
        try:
            event = events.get(timeout=4.0)
        except queue.Empty:
            # Keep browsers/reverse proxies alive while a model or Demucs is busy.
            yield ": heartbeat\n\n"
            continue
        if event is finished:
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def extract_cover(audio_path):
    try:
        audio = MutagenFile(audio_path)
        if audio is None:
            return None
        pictures = getattr(audio, "pictures", None) or []
        if pictures:
            picture = pictures[0]
            mime = getattr(picture, "mime", None) or "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(picture.data).decode('ascii')}"
        tags = getattr(audio, "tags", None)
        if tags:
            for tag in tags.values():
                if isinstance(tag, APIC):
                    return f"data:{tag.mime or 'image/jpeg'};base64,{base64.b64encode(tag.data).decode('ascii')}"
            covers = tags.get("covr") if hasattr(tags, "get") else None
            if covers:
                raw = bytes(covers[0])
                mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
                return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception as exc:
        logger.warning("Не удалось извлечь обложку: %s", exc)
    return None


def fetch_cover_online(artist, title):
    if not artist and not title:
        return None
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{artist} {title}".strip(), "entity": "song", "limit": 5},
            headers={"User-Agent": "LRCStudio/7.0"},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        cover = results[0].get("artworkUrl100") or results[0].get("artworkUrl60")
        return cover.replace("100x100bb", "600x600bb") if cover else None
    except Exception as exc:
        logger.warning("Не удалось найти обложку: %s", exc)
        return None


def get_artist_title(filename):
    name = Path(filename).stem
    name = name.split("(")[0].strip()
    name = name.split("_SkySound")[0].strip()
    if " - " in name:
        artist, title = name.split(" - ", 1)
        return artist.strip(), title.strip()
    if "_-_" in name:
        artist, title = name.split("_-_", 1)
        return artist.strip(), title.strip()
    return "", name


@app.route("/")
def index():
    html = render_template("index.html")
    # Inject V7 forced-alignment controls without forking the large legacy template.
    head = '<link rel="stylesheet" href="/static/v7-enhancer.css">'
    script = '<script src="/static/v7-enhancer.js"></script>'
    if head not in html:
        html = html.replace("</head>", head + "</head>")
    if script not in html:
        html = html.replace("</body>", script + "</body>")
    return html


@app.route("/cover", methods=["GET"])
def search_cover():
    artist = request.args.get("artist", "").strip()
    title = request.args.get("title", "").strip()
    cover = fetch_cover_online(artist, title)
    return jsonify({"cover": cover}) if cover else (jsonify({"error": "Обложка не найдена"}), 404)


@app.route("/search", methods=["GET"])
def search_lyrics():
    artist = request.args.get("artist", "").strip()
    title = request.args.get("title", "").strip()
    album = request.args.get("album", "").strip()
    from lrc_maker import iter_search_providers, parse_lrc

    def generate():
        for name, status, lyrics in iter_search_providers(artist, title, album):
            if status == "found":
                data = {
                    "type": "provider",
                    "source": name,
                    "status": "found",
                    "synced": bool(parse_lrc(lyrics)),
                    "lyrics": lyrics,
                }
            else:
                data = {"type": "provider", "source": name, "status": status}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield 'data: {"type":"done"}\n\n'

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.route("/parse-lyrics", methods=["POST"])
def parse_lyrics_file():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or ""
    if len(text.encode("utf-8")) > 2 * 1024 * 1024:
        return jsonify({"error": "Файл текста слишком большой"}), 413
    parsed = parse_lyrics(text)
    if not parsed["lines"]:
        return jsonify({"ok": True, "timed": False, "plain_lines": parsed["plain_lines"], "metadata": parsed["metadata"]})
    outputs = build_outputs(parsed["lines"], parsed["metadata"])
    return jsonify({"ok": True, "timed": True, "metadata": parsed["metadata"], **outputs})


def _save_upload(file_storage) -> tuple[str, str]:
    extension = Path(file_storage.filename or "").suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError("Поддерживаются MP3, WAV, FLAC, M4A, AAC, OGG и OPUS")
    safe_name = secure_filename(file_storage.filename) or f"audio{extension}"
    path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{safe_name}"
    file_storage.save(path)
    return str(path), safe_name


@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "Файл не выбран"}), 400
    try:
        audio_path, source_name = _save_upload(request.files["audio"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    cover = extract_cover(audio_path) or request.form.get("cover_url", "").strip() or None
    artist = request.form.get("artist", "").strip()
    title = request.form.get("title", "").strip()
    album = request.form.get("album", "").strip()
    if not artist and not title:
        artist, title = get_artist_title(source_name)
    if not cover and artist and title:
        cover = fetch_cover_online(artist, title)

    stream = _stream_elrc(
        audio_path,
        source_name,
        cover=cover,
        cleanup=True,
        lrc_text=request.form.get("lyrics", "").strip() or None,
        artist=artist,
        title=title,
        album=album,
        use_demucs=_parse_demucs(request.form.get("use_demucs"), "auto"),
        language=request.form.get("language", "rus").strip(),
        quality_mode=request.form.get("quality_mode", "max").strip(),
    )
    return Response(stream_with_context(stream), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/trackdl", methods=["POST"])
def trackdl():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введи название песни"}), 400
    if not (BASE_DIR / "node_modules" / "track-dl").exists():
        return jsonify({"error": "Не установлен track-dl. Выполни: npm install track-dl"}), 500

    result_file = DOWNLOADS_DIR / "trackdl_result.json"
    result_file.unlink(missing_ok=True)
    cmd = ["node", str(BASE_DIR / "trackdl_auto.js"), query]
    try:
        proc = subprocess.run(
            cmd,
            cwd=DOWNLOADS_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=420,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Таймаут: скачивание заняло слишком долго"}), 504
    if not result_file.exists():
        err = (proc.stderr or "")[-1200:]
        return jsonify({"error": f"Скачивание не удалось. {err}".strip()}), 500
    try:
        res = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Не удалось прочитать результат скачивания"}), 500
    if "error" in res:
        return jsonify({"error": res["error"]}), 500
    audio_path = DOWNLOADS_DIR / os.path.basename(res.get("file", ""))
    if not audio_path.exists():
        return jsonify({"error": "Файл не найден после скачивания"}), 500
    return jsonify({"ok": True, **res})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    file_name = (data.get("file") or "").strip()
    if not file_name:
        return jsonify({"error": "Нет файла"}), 400
    audio_path = (DOWNLOADS_DIR / os.path.basename(file_name)).resolve()
    if DOWNLOADS_DIR.resolve() not in audio_path.parents or not audio_path.exists():
        return jsonify({"error": "Файл не найден"}), 404

    stream = _stream_elrc(
        str(audio_path),
        file_name,
        cover=extract_cover(str(audio_path)),
        mp3=file_name,
        lrc_text=data.get("lyrics") or None,
        artist=(data.get("artist") or "").strip(),
        title=(data.get("title") or "").strip(),
        album=(data.get("album") or "").strip(),
        use_demucs=_parse_demucs(data.get("use_demucs"), "auto"),
        language=(data.get("language") or "rus").strip(),
        quality_mode=(data.get("quality_mode") or "max").strip(),
    )
    return Response(stream_with_context(stream), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/downloads/<path:filename>")
def downloads_file(filename):
    return send_from_directory(DOWNLOADS_DIR, filename)


# ---------------------------------------------------------------------------
# Stable local API for the future Better Lyrics integration
# ---------------------------------------------------------------------------

@app.route("/v1/health", methods=["GET", "OPTIONS"])
def v1_health():
    if request.method == "OPTIONS":
        return ("", 204)
    cache = AlignmentCache().info()
    return jsonify({
        "ok": True,
        "service": "LRC Extandator",
        "engineVersion": ALIGNMENT_ENGINE_VERSION,
        "cache": cache,
        "runtime": runtime_info(),
        "tokenRequired": True,
    })


@app.route("/v1/cache", methods=["GET", "DELETE", "OPTIONS"])
def v1_cache():
    if request.method == "OPTIONS":
        return ("", 204)
    if not _is_v1_authorized():
        return jsonify({"error": "unauthorized"}), 401
    cache = AlignmentCache()
    if request.method == "DELETE":
        return jsonify({"ok": True, "cleared": cache.clear()})
    return jsonify({"ok": True, **cache.info()})


@app.route("/v1/align", methods=["POST", "OPTIONS"])
def v1_align():
    if request.method == "OPTIONS":
        return ("", 204)
    if not _is_v1_authorized():
        return jsonify({"error": "unauthorized"}), 401
    if "audio" not in request.files:
        return jsonify({"error": "audio multipart field is required"}), 400
    try:
        audio_path, source_name = _save_upload(request.files["audio"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    stream = _stream_elrc(
        audio_path,
        source_name,
        cleanup=True,
        lrc_text=request.form.get("lyrics", "").strip() or None,
        artist=request.form.get("artist", "").strip(),
        title=request.form.get("title", "").strip(),
        album=request.form.get("album", "").strip(),
        language=request.form.get("language", "mul").strip(),
        use_demucs=_parse_demucs(request.form.get("use_demucs"), "auto"),
        quality_mode=request.form.get("quality_mode", "max").strip(),
        force_realign=request.form.get("force_realign", "false").lower() == "true",
    )
    return Response(stream_with_context(stream), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    host = os.environ.get("LRC_STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("LRC_STUDIO_PORT", "5000"))
    print(
        f"\nLRC Studio / Extandator Forced Alignment V7\nhttp://{host}:{port}\n"
        f"Local API token: {API_TOKEN}\n"
        "(token is also stored in .lrc_extandator/api-token.txt)\n"
    )
    app.run(debug=False, threaded=True, use_reloader=False, host=host, port=port)
