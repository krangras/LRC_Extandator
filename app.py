from flask import Flask, request, render_template, Response, jsonify, stream_with_context
import os
import traceback
import base64
import time
import json
import logging
import sys
import subprocess
from werkzeug.utils import secure_filename
from flask import send_from_directory
from lrc_maker import generate_elrc
from mutagen.id3 import APIC
from mutagen import File as MutagenFile
from pathlib import Path
import requests
import queue
import threading
import uuid
from lrc_formats import build_outputs, parse_lyrics

# Windows-консоль в cp1252/cp1251 не переваривает эмодзи и рамки баннера —
# переключаем stdout/stderr на UTF-8, иначе print()/логи падают.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# === НАСТРОЙКА ЛОГОВ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('lrc_studio.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus'}


def _result_payload(result, source_name, cover=None, mp3=None):
    if not result.get('success'):
        raise RuntimeError(result.get('error') or 'Не удалось синхронизировать текст')
    base_name = Path(source_name).stem
    payload = {
        'type': 'result',
        'elrc': result['elrc'],
        'elrc_compatible': result.get('elrc_compatible', result['elrc']),
        'lrc': result['lrc'],
        'filename': f'{base_name}.elrc',
        'lrc_filename': f'{base_name}.lrc',
        'cover': cover,
        'lines': result['lines'],
        'artist': result.get('artist', ''),
        'title': result.get('title', ''),
        'album': result.get('album', ''),
        'validation_issues': result.get('validation_issues', []),
        'alignment_quality': result.get('alignment_quality'),
    }
    if mp3:
        payload['mp3'] = mp3
    return payload


def _stream_elrc(audio_path, source_name, *, cover=None, cleanup=False, mp3=None, **options):
    """Run the heavy aligner in a worker and stream real progress as SSE."""
    events = queue.Queue()
    finished = object()

    def on_progress(percent, message):
        events.put({'type': 'progress', 'percent': percent, 'message': message})

    def worker():
        try:
            result = generate_elrc(audio_path, progress_callback=on_progress, **options)
            events.put(_result_payload(result, source_name, cover=cover, mp3=mp3))
        except Exception as exc:
            logger.error("Generation failed: %s", traceback.format_exc())
            events.put({'type': 'error', 'message': str(exc)})
        finally:
            if cleanup:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            events.put(finished)

    threading.Thread(target=worker, name='elrc-generator', daemon=True).start()
    while True:
        event = events.get()
        if event is finished:
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

# ===== ОБЛОЖКА =====
def extract_cover(audio_path):
    try:
        audio = MutagenFile(audio_path)
        if audio is None:
            return None
        pictures = getattr(audio, 'pictures', None) or []
        if pictures:
            picture = pictures[0]
            return f"data:{getattr(picture, 'mime', None) or 'image/jpeg'};base64,{base64.b64encode(picture.data).decode('ascii')}"
        tags = getattr(audio, 'tags', None)
        if tags:
            for tag in tags.values():
                if isinstance(tag, APIC):
                    return f"data:{tag.mime or 'image/jpeg'};base64,{base64.b64encode(tag.data).decode('ascii')}"
            covers = tags.get('covr') if hasattr(tags, 'get') else None
            if covers:
                raw = bytes(covers[0])
                mime = 'image/png' if raw.startswith(b'\x89PNG') else 'image/jpeg'
                return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception as e:
        logger.warning(f"⚠️ Не удалось извлечь обложку: {e}")
    return None

def fetch_cover_online(artist, title):
    """Find cover art through the public iTunes Search API (no API key)."""
    if not artist and not title:
        return None
    try:
        response = requests.get(
            'https://itunes.apple.com/search',
            params={'term': f'{artist} {title}'.strip(), 'entity': 'song', 'limit': 5},
            headers={'User-Agent': 'LRCStudio/5.0'},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get('results') or []
        if not results:
            return None
        cover = results[0].get('artworkUrl100') or results[0].get('artworkUrl60')
        return cover.replace('100x100bb', '600x600bb') if cover else None
    except Exception as exc:
        logger.warning('Не удалось найти обложку: %s', exc)
        return None

def get_artist_title(filename):
    name = Path(filename).stem
    name = name.split('(')[0].strip()
    name = name.split('_SkySound')[0].strip()
    if ' - ' in name:
        parts = name.split(' - ', 1)
        return parts[0].strip(), parts[1].strip()
    elif '_-_' in name:
        parts = name.split('_-_', 1)
        return parts[0].strip(), parts[1].strip()
    else:
        return '', name

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Small endpoint used by the UI to distinguish a network failure from a job failure."""
    return jsonify({'ok': True})


@app.route('/cover', methods=['GET'])
def search_cover():
    artist = request.args.get('artist', '').strip()
    title = request.args.get('title', '').strip()
    cover = fetch_cover_online(artist, title)
    return jsonify({'cover': cover}) if cover else (jsonify({'error': 'Обложка не найдена'}), 404)

@app.route('/search', methods=['GET'])
def search_lyrics():
    """Поиск текста по всем источникам в реальном времени (SSE).

    Для каждого провайдера (LRCLIB, BetterLyrics, Musixmatch, NetEase) шлёт
    событие 'provider' со статусом searching/found/empty/error, в конце — 'done'."""
    artist = request.args.get('artist', '').strip()
    title = request.args.get('title', '').strip()
    album = request.args.get('album', '').strip()

    from lrc_maker import iter_search_providers, parse_lrc

    def generate():
        for name, status, lyrics in iter_search_providers(artist, title, album):
            if status == 'found':
                data = json.dumps({
                    'type': 'provider',
                    'source': name,
                    'status': 'found',
                    'synced': bool(parse_lrc(lyrics)),
                    'lyrics': lyrics,
                })
            else:
                data = json.dumps({
                    'type': 'provider',
                    'source': name,
                    'status': status,
                })
            yield f"data: {data}\n\n"
        yield "data: {\"type\":\"done\"}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route('/parse-lyrics', methods=['POST'])
def parse_lyrics_file():
    data = request.get_json(silent=True) or {}
    text = data.get('text') or ''
    if len(text.encode('utf-8')) > 2 * 1024 * 1024:
        return jsonify({'error': 'Файл текста слишком большой'}), 413
    parsed = parse_lyrics(text)
    if not parsed['lines']:
        return jsonify({
            'ok': True,
            'timed': False,
            'plain_lines': parsed['plain_lines'],
            'metadata': parsed['metadata'],
        })
    outputs = build_outputs(parsed['lines'], parsed['metadata'])
    return jsonify({
        'ok': True,
        'timed': True,
        'metadata': parsed['metadata'],
        **outputs,
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({'error': 'Нет аудиофайла'}), 400
    
    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    extension = Path(audio.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({'error': 'Поддерживаются MP3, WAV, FLAC, M4A, AAC, OGG и OPUS'}), 400
    safe_name = secure_filename(audio.filename) or f'audio{extension}'
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uuid.uuid4().hex}_{safe_name}')
    audio.save(audio_path)
    
    cover = extract_cover(audio_path)
    if not cover:
        cover_url = request.form.get('cover_url', '').strip()
        if cover_url:
            cover = cover_url
    
    artist = request.form.get('artist', '').strip()
    title = request.form.get('title', '').strip()
    album = request.form.get('album', '').strip()
    if not artist and not title:
        artist, title = get_artist_title(audio.filename)
    
    if not cover and artist and title:
        cover = fetch_cover_online(artist, title)
    
    lyrics_text = request.form.get('lyrics', '').strip()
    language = request.form.get('language', 'rus').strip()
    use_demucs = request.form.get('use_demucs', 'true').lower() == 'true'

    logger.info("🚀 ===== НАЧАЛО ОБРАБОТКИ =====")
    logger.info("📁 Файл: %s", audio.filename)
    stream = _stream_elrc(
        audio_path,
        audio.filename,
        cover=cover,
        cleanup=True,
        lrc_text=lyrics_text or None,
        artist=artist,
        title=title,
        album=album,
        use_demucs=use_demucs,
        language=language,
    )
    return Response(stream_with_context(stream), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })

# ===== СКАЧИВАНИЕ (track-dl) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR = os.path.join(BASE_DIR, 'downloads')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

@app.route('/trackdl', methods=['POST'])
def trackdl():
    """Скачивание трека через track-dl (YouTube + метаданные Deezer/iTunes).

    Запускает trackdl_auto.js (без интерактива), который пишет результат
    в trackdl_result.json внутри downloads/. Возвращает имя файла и метаданные.
    """
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'error': 'Введи название песни'}), 400

    if not os.path.exists(os.path.join(BASE_DIR, 'node_modules', 'track-dl')):
        return jsonify({'error': 'Не установлен track-dl. Выполни в папке проекта: npm install track-dl'}), 500

    result_file = os.path.join(DOWNLOADS_DIR, 'trackdl_result.json')
    if os.path.exists(result_file):
        os.remove(result_file)

    cmd = ['node', os.path.join(BASE_DIR, 'trackdl_auto.js'), query]
    try:
        proc = subprocess.run(
            cmd,
            cwd=DOWNLOADS_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=420,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Таймаут: скачивание заняло слишком долго'}), 504

    if not os.path.exists(result_file):
        err = (proc.stderr or '')[-1200:]
        detail = f' {err}' if err else ''
        return jsonify({'error': f'Скачивание не удалось.{detail}'}), 500

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            res = json.load(f)
    except Exception:
        return jsonify({'error': 'Не удалось прочитать результат скачивания'}), 500

    if 'error' in res:
        return jsonify({'error': res['error']}), 500

    audio_path = os.path.join(DOWNLOADS_DIR, os.path.basename(res.get('file', '')))
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Файл не найден после скачивания'}), 500

    return jsonify({'ok': True, **res})

@app.route('/generate', methods=['POST'])
def generate():
    """Генерация ELRC из файла, скачанного через track-dl (лежит в downloads/)."""
    data = request.get_json(silent=True) or {}
    file_name = (data.get('file') or '').strip()
    if not file_name:
        return jsonify({'error': 'Нет файла'}), 400

    audio_path = os.path.join(DOWNLOADS_DIR, os.path.basename(file_name))
    if not os.path.abspath(audio_path).startswith(os.path.abspath(DOWNLOADS_DIR)) or not os.path.exists(audio_path):
        return jsonify({'error': 'Файл не найден'}), 404

    artist = (data.get('artist') or '').strip()
    title = (data.get('title') or '').strip()
    album = (data.get('album') or '').strip()
    language = (data.get('language') or 'rus').strip()
    use_demucs = bool(data.get('use_demucs', True))
    cover = extract_cover(audio_path)

    logger.info("🎬 ===== ГЕНЕРАЦИЯ ИЗ СКАЧАННОГО ФАЙЛА =====")
    stream = _stream_elrc(
        audio_path,
        file_name,
        cover=cover,
        mp3=file_name,
        lrc_text=None,
        artist=artist,
        title=title,
        album=album,
        use_demucs=use_demucs,
        language=language,
    )
    return Response(stream_with_context(stream), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })

@app.route('/downloads/<path:filename>')
def downloads_file(filename):
    return send_from_directory(DOWNLOADS_DIR, filename)

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════╗
    ║   LRC STUDIO v5.0                    ║
    ║   Полный ELRC + точный редактор      ║
    ║   http://127.0.0.1:5000             ║
    ╚═══════════════════════════════════════╝
    """)
    debug = os.environ.get('LRC_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    host = os.environ.get('LRC_HOST', '127.0.0.1')
    port = int(os.environ.get('LRC_PORT', '5000'))
    app.run(debug=debug, threaded=True, use_reloader=False, host=host, port=port)
