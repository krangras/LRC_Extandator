(() => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const lyricsInput = document.getElementById('lyricsInput');
    const artistInput = document.getElementById('artistInput');
    const titleInput = document.getElementById('titleInput');
    const albumInput = document.getElementById('albumInput');
    const charCount = document.getElementById('charCount');
    const languageSelect = document.getElementById('languageSelect');
    const formatSelect = document.getElementById('formatSelect');
    const generateBtn = document.getElementById('generateBtn');
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressLabel = document.getElementById('progressLabel');
    const progressPercent = document.getElementById('progressPercent');
    const resultSection = document.getElementById('resultSection');
    const resultTitleText = document.getElementById('resultTitleText');
    const copyBtn = document.getElementById('copyBtn');
    const editBtn = document.getElementById('editBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const downloadSimpleBtn = document.getElementById('downloadSimpleBtn');
    const lrcOutput = document.getElementById('lrcOutput');
    const toast = document.getElementById('toast');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');

    const playerContainer = document.getElementById('playerContainer');
    const playBtn = document.getElementById('playBtn');
    const seekSlider = document.getElementById('seekSlider');
    const timeDisplay = document.getElementById('timeDisplay');
    const karaokeContainer = document.getElementById('karaokeContainer');

    const logContainer = document.getElementById('logContainer');

    let currentLrc = '';
    let currentFilename = '';
    let currentLrcFilename = '';
    let currentOutputs = { elrc: '', elrcCompatible: '', lrc: '' };
    let currentMetadata = {};
    let fileToProcess = null;
    let audioPlayer = null;
    let animationId = null;

    // ===== localStorage =====
    const savedLanguage = localStorage.getItem('lrc_language');
    const savedFormat = localStorage.getItem('lrc_format');
    if (savedLanguage) languageSelect.value = savedLanguage;
    if (savedFormat && [...formatSelect.options].some(option => option.value === savedFormat)) formatSelect.value = savedFormat;
    languageSelect.addEventListener('change', () => localStorage.setItem('lrc_language', languageSelect.value));
    formatSelect.addEventListener('change', () => localStorage.setItem('lrc_format', formatSelect.value));

    // ===== ФОРМАТИРОВАНИЕ ВРЕМЕНИ =====
    function formatTimestamp(seconds) {
        const value = Number(seconds);
        if (!Number.isFinite(value)) return '00:00.000';
        const totalMs = Math.max(0, Math.round(value * 1000));
        const m = Math.floor(totalMs / 60000);
        const s = Math.floor((totalMs % 60000) / 1000);
        const ms = totalMs % 1000;
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
    }

    function parseTimestamp(str) {
        const match = String(str).match(/[\[<]?(\d{1,4}):([0-5]?\d)(?:[\.:](\d{1,3}))?[\]>]?/);
        if (match) {
            const m = parseInt(match[1]);
            const s = parseInt(match[2]);
            const fraction = match[3] || '';
            return m * 60 + s + (fraction ? parseInt(fraction) / Math.pow(10, fraction.length) : 0);
        }
        return NaN;
    }

    async function consumeSSEResponse(response, onEvent) {
        if (!response.ok) {
            let message = `Ошибка сервера (${response.status})`;
            try { message = (await response.json()).error || message; } catch (_) {}
            throw new Error(message);
        }
        if (!response.body) throw new Error('Браузер не поддерживает потоковый ответ');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            let boundary;
            while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                const block = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                for (const line of block.split('\n')) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (payload) onEvent(JSON.parse(payload));
                }
            }
            if (done) break;
        }
    }

    // ===== ЧТЕНИЕ ТЕГОВ MP3 (ID3v2 + ID3v1) =====
    function synchsafe(a, b, c, d) {
        return ((a & 0x7f) << 21) | ((b & 0x7f) << 14) | ((c & 0x7f) << 7) | (d & 0x7f);
    }

    function decodeCp1251(bytes) {
        let s = '';
        for (let i = 0; i < bytes.length; i++) {
            const b = bytes[i];
            let c;
            if (b < 0x80) c = b;
            else if (b >= 0xC0 && b <= 0xFF) c = 0x0410 + (b - 0xC0);
            else if (b === 0xA8) c = 0x0401; // Ё
            else if (b === 0xB8) c = 0x0451; // ё
            else c = 0x2500 + (b - 0x80);
            s += String.fromCharCode(c);
        }
        return s;
    }

    function cleanTagText(s) {
        const parts = s.split('\u0000').map(p => p.trim()).filter(Boolean);
        return parts[0] || '';
    }

    function decodeId3Text(bytes, start, size) {
        try {
            const enc = bytes[start];
            let text = '';
            if (enc === 1) { // UTF-16 с BOM
                const b1 = bytes[start + 1], b2 = bytes[start + 2];
                const dec = new TextDecoder((b1 === 0xff && b2 === 0xfe) ? 'utf-16le' : 'utf-16be');
                text = dec.decode(bytes.subarray(start + 3, start + size));
            } else if (enc === 2) { // UTF-16BE
                text = new TextDecoder('utf-16be').decode(bytes.subarray(start + 1, start + size));
            } else if (enc === 3) { // UTF-8
                text = new TextDecoder('utf-8').decode(bytes.subarray(start + 1, start + size));
            } else { // 0 = ISO-8859-1 / CP1251 (эвристика)
                const raw = bytes.subarray(start + 1, start + size);
                const cp = decodeCp1251(raw);
                let cyr = 0;
                for (let i = 0; i < cp.length; i++) {
                    const cc = cp.charCodeAt(i);
                    if (cc >= 0x0400 && cc <= 0x04FF) cyr++;
                }
                if (cyr > 0 && cyr >= cp.length * 0.3) {
                    text = cp;
                } else {
                    text = new TextDecoder('windows-1252').decode(raw);
                }
            }
            return cleanTagText(text);
        } catch (e) {
            return '';
        }
    }

    function parseId3v2(bytes) {
        if (bytes.length < 10 || bytes[0] !== 0x49 || bytes[1] !== 0x44 || bytes[2] !== 0x33) return null;
        const version = bytes[3];
        const flags = bytes[5];
        const tagSize = synchsafe(bytes[6], bytes[7], bytes[8], bytes[9]);
        let pos = 10;
        if (flags & 0x40) { // extended header
            if (version === 3) {
                const extSize = (bytes[10] << 24) | (bytes[11] << 16) | (bytes[12] << 8) | bytes[13];
                pos = 10 + 4 + extSize;
            } else {
                const extSize = synchsafe(bytes[10], bytes[11], bytes[12], bytes[13]);
                pos = 10 + 4 + extSize;
            }
        }
        const end = Math.min(bytes.length, 10 + tagSize);
        const frameIdLen = version === 2 ? 3 : 4;
        const map = version === 2
            ? { TT2: 'title', TP1: 'artist', TAL: 'album' }
            : { TIT2: 'title', TPE1: 'artist', TALB: 'album' };
        const out = {};
        while (pos + frameIdLen <= end) {
            let frameId = '';
            for (let i = 0; i < frameIdLen; i++) frameId += String.fromCharCode(bytes[pos + i]);
            if (!/^[A-Z0-9]{3,4}$/.test(frameId)) break;
            let frameSize, headerLen;
            if (version === 2) {
                frameSize = (bytes[pos + 3] << 16) | (bytes[pos + 4] << 8) | bytes[pos + 5];
                headerLen = 6;
            } else if (version === 4) {
                frameSize = synchsafe(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
                headerLen = 10;
            } else {
                frameSize = (bytes[pos + 4] << 24) | (bytes[pos + 5] << 16) | (bytes[pos + 6] << 8) | bytes[pos + 7];
                headerLen = 10;
            }
            if (frameSize <= 0 || pos + headerLen + frameSize > end) break;
            const dataStart = pos + headerLen;
            const key = map[frameId];
            if (key && !out[key]) {
                out[key] = decodeId3Text(bytes, dataStart, frameSize);
            }
            pos = dataStart + frameSize;
        }
        return out;
    }

    function readMp3Tags(file) {
        return new Promise((resolve) => {
            const defaults = { artist: '', title: '', album: '' };
            if (!file) return resolve(defaults);
            const result = { ...defaults };
            const head = file.slice(0, 256 * 1024);
            const tail = file.slice(Math.max(0, file.size - 128));
            let headDone = false, tailDone = false;
            const finish = () => { if (headDone && tailDone) resolve(result); };
            const headReader = new FileReader();
            const tailReader = new FileReader();
            headReader.onerror = () => { headDone = true; finish(); };
            tailReader.onerror = () => { tailDone = true; finish(); };
            headReader.onload = () => {
                try {
                    const tags = parseId3v2(new Uint8Array(headReader.result));
                    if (tags) {
                        if (tags.artist) result.artist = tags.artist;
                        if (tags.title) result.title = tags.title;
                        if (tags.album) result.album = tags.album;
                    }
                } catch (e) {}
                headDone = true;
                finish();
            };
            tailReader.onload = () => {
                try {
                    const bytes = new Uint8Array(tailReader.result);
                    if (bytes.length >= 128 && String.fromCharCode(bytes[0], bytes[1], bytes[2]) === 'TAG') {
                        const txt = (off) => String.fromCharCode.apply(null, bytes.subarray(off, off + 30)).replace(/\u0000/g, '').trim();
                        if (!result.title) result.title = txt(3);
                        if (!result.artist) result.artist = txt(33);
                        if (!result.album) result.album = txt(63);
                    }
                } catch (e) {}
                tailDone = true;
                finish();
            };
            headReader.readAsArrayBuffer(head);
            tailReader.readAsArrayBuffer(tail);
        });
    }

    // ===== DROP ZONE =====
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        const extension = file.name.toLowerCase().split('.').pop();
        if (!['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg', 'opus'].includes(extension)) {
            showToast('Выбери аудиофайл: MP3, WAV, FLAC, M4A, AAC, OGG или OPUS', 'error');
            return;
        }
        if (file.size > 200 * 1024 * 1024) {
            showToast('Файл слишком большой (макс. 200 МБ)', 'error');
            return;
        }
        fileToProcess = file;
        dropZone.classList.add('has-file');
        dropZone.innerHTML = `
            <div class="file-info">
                <div class="file-icon"><svg viewBox="0 0 24 24"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg></div>
                <span class="file-name">${file.name}</span>
                <button class="file-remove" onclick="event.stopPropagation(); resetDropZone()">
                    <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>
        `;
        dropZone.onclick = null;
        document.getElementById('coverContainer').classList.add('hidden');
        playerContainer.classList.add('hidden');
        if (audioPlayer) { audioPlayer.pause(); audioPlayer = null; }
        if (animationId) cancelAnimationFrame(animationId);

        readMp3Tags(file).then((tags) => {
            if (tags.artist) artistInput.value = tags.artist;
            if (tags.title) titleInput.value = tags.title;
            if (tags.album) albumInput.value = tags.album;
            addLog(`🎤 Исполнитель: ${tags.artist || '—'}`);
            addLog(`🎵 Название: ${tags.title || '—'}`);
            addLog(`💿 Альбом: ${tags.album || '—'}`);
            if (tags.artist || tags.title) {
                startLyricsSearch(tags.artist, tags.title, tags.album);
            }
        });
    }

    window.resetDropZone = function () {
        fileToProcess = null;
        dropZone.classList.remove('has-file');
        dropZone.innerHTML = `
            <div class="drop-icon"><svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></div>
            <p class="drop-text">Перетащи аудио сюда или <strong>выбери файл</strong></p>
            <span class="drop-hint">MP3, WAV, FLAC, M4A, AAC, OGG, OPUS · до 200 МБ</span>
        `;
        dropZone.onclick = () => fileInput.click();
        fileInput.value = '';
        document.getElementById('coverContainer').classList.add('hidden');
        playerContainer.classList.add('hidden');
        if (audioPlayer) { audioPlayer.pause(); audioPlayer = null; }
        if (animationId) cancelAnimationFrame(animationId);
    };

    const coverUrlInput = document.getElementById('coverUrlInput');
    const coverUrlBtn = document.getElementById('coverUrlBtn');
    const coverSearchBtn = document.getElementById('coverSearchBtn');
    function showManualCover(url, source) {
        if (!/^https?:\/\//i.test(url) && !/^data:image\//i.test(url)) throw new Error('Нужна прямая ссылка на изображение');
        const image = document.getElementById('coverImage');
        image.onerror = () => showToast('Обложка по этой ссылке не загрузилась', 'error');
        image.src = url;
        document.getElementById('coverTitle').textContent = titleInput.value.trim() || 'Без названия';
        document.getElementById('coverArtist').textContent = artistInput.value.trim() || 'Исполнитель не указан';
        document.getElementById('coverSource').textContent = source;
        document.getElementById('coverContainer').classList.remove('hidden');
    }
    coverUrlBtn.addEventListener('click', () => {
        try { showManualCover(coverUrlInput.value.trim(), 'ручная ссылка'); }
        catch (error) { showToast(error.message, 'error'); }
    });
    coverSearchBtn.addEventListener('click', async () => {
        const params = new URLSearchParams({ artist: artistInput.value.trim(), title: titleInput.value.trim() });
        coverSearchBtn.disabled = true;
        try {
            const response = await fetch('/cover?' + params);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Обложка не найдена');
            coverUrlInput.value = data.cover;
            showManualCover(data.cover, 'iTunes Search');
        } catch (error) { showToast(error.message, 'error'); }
        finally { coverSearchBtn.disabled = false; }
    });

    lyricsInput.addEventListener('input', () => {
        charCount.textContent = lyricsInput.value.length;
    });

    // ===== ПОИСК ТЕКСТА (живые результаты по источникам) =====
    const searchLyricsBtn = document.getElementById('searchLyricsBtn');
    const searchResults = document.getElementById('searchResults');
    const searchResultsList = document.getElementById('searchResultsList');
    const searchResultsTitle = document.getElementById('searchResultsTitle');
    const closeSearchBtn = document.getElementById('closeSearchBtn');

    let providerRows = {};

    function providerStateLabel(status) {
        switch (status) {
            case 'searching': return '⏳ ищем...';
            case 'found': return '✅ найдено';
            case 'empty': return '— не найдено';
            case 'error': return '❌ ошибка';
            default: return '';
        }
    }

    function ensureProviderRow(name) {
        if (providerRows[name]) return providerRows[name];
        const row = document.createElement('div');
        row.className = 'search-provider-row';
        row.innerHTML = `<span class="search-provider-name">${name}</span><span class="search-provider-state"></span>`;
        searchResultsList.appendChild(row);
        providerRows[name] = row;
        return row;
    }

    function setProviderState(name, status) {
        const row = ensureProviderRow(name);
        const stateEl = row.querySelector('.search-provider-state');
        stateEl.textContent = providerStateLabel(status);
        stateEl.className = 'search-provider-state ' + status;
    }

    function addSearchResult(data) {
        setProviderState(data.source, 'found');
        const card = document.createElement('div');
        card.className = 'search-result';
        card.dataset.source = data.source;

        const isSynced = !!data.synced;
        const head = document.createElement('div');
        head.className = 'search-result-head';
        head.innerHTML = `
            <span class="search-result-source">${data.source}</span>
            <span class="search-result-status ${isSynced ? 'synced' : ''}">${isSynced ? '✅ синхронизированный LRC' : 'ℹ️ текст без таймингов'}</span>
            <div class="search-result-actions">
                <button type="button" class="btn tiny" data-act="use">Использовать</button>
                <button type="button" class="btn tiny" data-act="copy">Копировать</button>
            </div>
        `;

        const pre = document.createElement('pre');
        pre.className = 'search-result-lyrics';
        pre.textContent = data.lyrics;

        card.appendChild(head);
        card.appendChild(pre);
        searchResultsList.appendChild(card);

        head.querySelector('[data-act="use"]').addEventListener('click', () => {
            lyricsInput.value = data.lyrics;
            charCount.textContent = data.lyrics.length;
            showToast('Текст вставлен в поле', 'success');
        });
        head.querySelector('[data-act="copy"]').addEventListener('click', () => {
            navigator.clipboard.writeText(data.lyrics).then(() => showToast('Текст скопирован', 'success'));
        });
    }

    function startLyricsSearch(artist, title, album) {
        providerRows = {};
        searchResultsList.innerHTML = '';
        searchResultsTitle.textContent = `Поиск текста: ${[artist, title].filter(Boolean).join(' — ') || 'по запросу'}`;
        searchResults.classList.remove('hidden');
        searchLyricsBtn.disabled = true;
        searchLyricsBtn.textContent = '⏳ Поиск...';

        const params = new URLSearchParams();
        if (artist) params.set('artist', artist);
        if (title) params.set('title', title);
        if (album) params.set('album', album);

        fetch('/search?' + params.toString())
            .then(res => {
                if (!res.ok) throw new Error('Ошибка сервера');
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                const pump = () => reader.read().then(({ done, value }) => {
                    if (done) {
                        searchResultsTitle.textContent = 'Результаты поиска';
                        searchLyricsBtn.disabled = false;
                        searchLyricsBtn.textContent = '🔍 Найти текст';
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n')) !== -1) {
                        const line = buffer.slice(0, idx).trim();
                        buffer = buffer.slice(idx + 1);
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'provider') {
                                    if (data.status === 'found') {
                                        addSearchResult(data);
                                    } else {
                                        setProviderState(data.source, data.status);
                                    }
                                }
                            } catch (e) {}
                        }
                    }
                    return pump();
                });
                return pump();
            })
            .catch(err => {
                searchLyricsBtn.disabled = false;
                searchLyricsBtn.textContent = '🔍 Найти текст';
                showToast('Ошибка поиска: ' + err.message, 'error');
            });
    }

    searchLyricsBtn.addEventListener('click', () => {
        const artist = artistInput.value.trim();
        const title = titleInput.value.trim();
        if (!artist && !title) {
            showToast('Сначала укажи исполнителя или название', 'error');
            return;
        }
        startLyricsSearch(artist, title, albumInput.value.trim());
    });

    closeSearchBtn.addEventListener('click', () => {
        searchResults.classList.add('hidden');
    });

    // ===== ИМПОРТ ГОТОВОГО LRC / ELRC =====
    const importLyricsBtn = document.getElementById('importLyricsBtn');
    const lyricsFileInput = document.getElementById('lyricsFileInput');
    importLyricsBtn.addEventListener('click', () => lyricsFileInput.click());
    lyricsFileInput.addEventListener('change', async () => {
        const file = lyricsFileInput.files && lyricsFileInput.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            lyricsInput.value = text;
            charCount.textContent = text.length;
            const response = await fetch('/parse-lyrics', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            if (!response.ok || data.error) throw new Error(data.error || 'Не удалось разобрать файл');
            if (data.metadata) {
                artistInput.value = data.metadata.ar || artistInput.value;
                titleInput.value = data.metadata.ti || titleInput.value;
                albumInput.value = data.metadata.al || albumInput.value;
            }
            if (!data.timed) {
                showToast('Текст загружен. Запусти синхронизацию с аудио.', 'success');
                return;
            }
            currentMetadata = data.metadata || {};
            currentOutputs = {
                elrc: data.elrc,
                elrcCompatible: data.elrc_compatible,
                lrc: data.lrc
            };
            currentLrc = selectOutputByFormat();
            currentFilename = file.name.replace(/\.(?:lrc|elrc|txt)$/i, '.elrc');
            currentLrcFilename = file.name.replace(/\.(?:lrc|elrc|txt)$/i, '.lrc');
            refreshFormatUi();
            resultSection.classList.add('visible');
            window.lastGeneratedLines = data.lines;
            if (fileToProcess) {
                document.getElementById('editorTabBtn').disabled = false;
                openEditor(data.lines, fileToProcess);
                document.querySelector('[data-target="editorTab"]').click();
                showToast('ELRC открыт в редакторе', 'success');
            } else {
                showToast('ELRC разобран. Добавь аудио для точного редактирования.', 'success');
            }
        } catch (error) {
            showToast('Ошибка импорта: ' + error.message, 'error');
        } finally {
            lyricsFileInput.value = '';
        }
    });

    // ===== ГЕНЕРАЦИЯ =====
    generateBtn.addEventListener('click', () => {
        if (!fileToProcess) {
            showToast('Сначала выбери аудиофайл', 'error');
            return;
        }
        startProcessing(fileToProcess);
    });

    function startProcessing(file) {
        if (window.location.protocol === 'file:') {
            showToast('Открой приложение через http://127.0.0.1:5000, а не напрямую как HTML-файл', 'error');
            addLog('❌ Страница запущена через file://. Запусти ./run.sh и открой http://127.0.0.1:5000');
            return;
        }
        const formData = new FormData();
        formData.append('audio', file);
        formData.append('lyrics', lyricsInput.value);
        formData.append('language', languageSelect.value);
        formData.append('format', formatSelect.value);
        formData.append('artist', artistInput.value.trim());
        formData.append('title', titleInput.value.trim());
        formData.append('album', albumInput.value.trim());
        formData.append('cover_url', coverUrlInput.value.trim());

        progressFill.style.width = '0%';
        progressLabel.textContent = 'Подготовка...';
        progressPercent.textContent = '0%';
        [step1, step2, step3].forEach(s => s.classList.remove('active', 'done'));
        logContainer.innerHTML = '';
        logContainer.style.display = 'block';
        addLog('⏳ Начинаем обработку...');

        generateBtn.disabled = true;
        generateBtn.classList.add('loading');
        generateBtn.querySelector('.btn-text').textContent = 'Обработка...';
        progressSection.classList.add('visible');
        resultSection.classList.remove('visible');
        playerContainer.classList.add('hidden');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(async res => {
            let resultData = null;
            await consumeSSEResponse(res, (data) => {
                if (data.type === 'progress') {
                    updateProgress(data.percent, data.message);
                    addLog(`▸ ${data.message} (${data.percent}%)`);
                } else if (data.type === 'result') {
                    resultData = data;
                } else if (data.type === 'error') {
                    throw new Error(data.message);
                }
            });
            if (resultData) {
                applyResult(resultData, fileToProcess);
            } else {
                throw new Error('Сервер не вернул результат');
            }
        })
        .catch(err => {
            const detail = err && err.message === 'Failed to fetch'
                ? 'Сервер недоступен или соединение оборвалось. Проверь http://127.0.0.1:5000/health и терминал с ./run.sh.'
                : (err.message || String(err));
            showToast('Ошибка: ' + detail, 'error');
            resetUI();
            addLog('❌ Ошибка: ' + detail);
        });
    }

    // ===== ОБЩИЙ РЕНДЕРИНГ РЕЗУЛЬТАТА (upload и Telegram) =====
    function applyResult(resultData, audioFile) {
        setStepDone(step3);
        if (audioFile) fileToProcess = audioFile;
        currentOutputs = {
            elrc: resultData.elrc || '',
            elrcCompatible: resultData.elrc_compatible || resultData.elrc || '',
            lrc: resultData.lrc || ''
        };
        currentMetadata = {
            ar: resultData.artist || '', ti: resultData.title || '', al: resultData.album || '',
            by: 'LRC Studio', re: 'LRC Studio', ve: '5.0'
        };
        currentLrc = selectOutputByFormat();
        currentFilename = resultData.filename;
        currentLrcFilename = resultData.lrc_filename || resultData.filename.replace(/\.elrc$/i, '.lrc');

        const coverContainer = document.getElementById('coverContainer');
        const coverImage = document.getElementById('coverImage');
        const coverTitle = document.getElementById('coverTitle');
        const coverArtist = document.getElementById('coverArtist');
        if (resultData.cover) {
            coverImage.src = resultData.cover;
            coverTitle.textContent = resultData.title || (audioFile ? audioFile.name.replace('.mp3', '') : '');
            coverArtist.textContent = [resultData.artist, resultData.album].filter(Boolean).join(' — ') || 'Из MP3-тегов';
            coverContainer.classList.remove('hidden');
            addLog('🖼️ Обложка загружена');
        } else {
            coverContainer.classList.add('hidden');
            addLog('ℹ️ Обложка не найдена в MP3');
        }

        refreshFormatUi();
        resultSection.classList.add('visible');

        if (resultData.lines && resultData.lines.length > 0 && audioFile) {
            setupPlayer(resultData.lines, audioFile);
            window.lastGeneratedLines = resultData.lines;
            document.getElementById('editorTabBtn').disabled = false;
        }

        resetUI();
        const issueCount = (resultData.validation_issues || []).length;
        const quality = Number(resultData.alignment_quality);
        if (Number.isFinite(quality)) {
            const percent = Math.round(Math.max(0, Math.min(1, quality)) * 100);
            addLog(`📊 Прямое сопоставление слов: ${percent}%`);
        }
        addLog(issueCount ? `⚠️ Готово, автоматически исправлено замечаний: ${issueCount}` : '✅ Готово! ELRC проверен и готов к скачиванию');
    }

    function selectOutputByFormat() {
        if (formatSelect.value === 'lrc') return currentOutputs.lrc || currentOutputs.elrc;
        if (formatSelect.value === 'elrc_compatible') return currentOutputs.elrcCompatible || currentOutputs.elrc;
        return currentOutputs.elrc;
    }

    function refreshFormatUi() {
        currentLrc = selectOutputByFormat();
        lrcOutput.textContent = currentLrc;
        const simple = formatSelect.value === 'lrc';
        resultTitleText.textContent = simple ? currentLrcFilename : currentFilename;
        downloadBtn.textContent = simple
            ? '💾 Скачать LRC'
            : (formatSelect.value === 'elrc_compatible' ? '💾 Скачать совместимый ELRC' : '💾 Скачать точный ELRC');
    }

    formatSelect.addEventListener('change', () => {
        if (!currentOutputs.elrc && !currentOutputs.lrc) return;
        refreshFormatUi();
    });

    // ===== ПЛЕЕР =====
    function setupPlayer(lines, file) {
        playerContainer.classList.remove('hidden');
        karaokeContainer.innerHTML = '';

        if (audioPlayer) {
            audioPlayer.pause();
            audioPlayer = null;
        }
        if (animationId) cancelAnimationFrame(animationId);

        audioPlayer = new Audio(URL.createObjectURL(file));
        let activeIndex = -1;

        lines.forEach((line, idx) => {
            const div = document.createElement('div');
            div.className = 'line';
            div.dataset.index = idx;
            div.dataset.start = line.start || 0;

            if (line.words && line.words.length > 0) {
                line.words.forEach((w) => {
                    const span = document.createElement('span');
                    span.className = 'word';
                    span.textContent = w.word + ' ';
                    span.dataset.start = w.start || 0;
                    span.dataset.end = w.end || 0;
                    div.appendChild(span);
                });
            } else {
                div.textContent = line.text || '';
            }
            karaokeContainer.appendChild(div);
        });

        function formatTime(seconds) {
            if (!seconds || isNaN(seconds)) return '00:00';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }

        function updateUI() {
            if (!audioPlayer) return;
            const current = audioPlayer.currentTime;
            const duration = audioPlayer.duration || 0;
            timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
            seekSlider.value = duration ? (current / duration) * 100 : 0;

            const lineElements = karaokeContainer.querySelectorAll('.line');
            let newActive = -1;
            for (let i = 0; i < lineElements.length; i++) {
                const start = parseFloat(lineElements[i].dataset.start);
                const nextStart = (i + 1 < lineElements.length) ? parseFloat(lineElements[i+1].dataset.start) : Infinity;
                if (current >= start && current < nextStart) {
                    newActive = i;
                    break;
                }
            }
            if (newActive !== activeIndex) {
                if (activeIndex !== -1) lineElements[activeIndex]?.classList.remove('active');
                if (newActive !== -1) {
                    lineElements[newActive].classList.add('active');
                    lineElements[newActive].scrollIntoView({ block: 'center', behavior: 'smooth' });
                }
                activeIndex = newActive;
            }

            if (activeIndex !== -1) {
                if (lineElements[activeIndex]) {
                    const lineStart = parseFloat(lineElements[activeIndex].dataset.start) || 0;
                    const words = lineElements[activeIndex].querySelectorAll('.word');
                    words.forEach(w => {
                        let start = parseFloat(w.dataset.start);
                        let end = parseFloat(w.dataset.end);
                        if (isNaN(start) || start < 0) start = lineStart;
                        if (isNaN(end) || end <= start) end = start + 1;
                        const progress = (current - start) / (end - start);
                        if (progress >= 1) {
                            w.style.color = '#a78bfa';
                            w.style.fontWeight = '700';
                        } else if (progress >= 0) {
                            w.style.color = '#e0e0e0';
                            w.style.fontWeight = '600';
                        } else {
                            w.style.color = '#555577';
                            w.style.fontWeight = '400';
                        }
                    });
                }
            }

            // Words outside their line's active window can still be sung: mark
            // every word by its own timestamp so none stays unmarked/white.
            lineElements.forEach(el => {
                const lineStart = parseFloat(el.dataset.start) || 0;
                const words = el.querySelectorAll('.word');
                words.forEach(w => {
                    let start = parseFloat(w.dataset.start);
                    let end = parseFloat(w.dataset.end);
                    if (isNaN(start) || start < 0) start = lineStart;
                    if (isNaN(end) || end <= start) end = start + 1;
                    const progress = (current - start) / (end - start);
                    if (progress >= 1 && w.style.color !== '#a78bfa') {
                        w.style.color = '#a78bfa';
                        w.style.fontWeight = '700';
                    } else if (progress >= 0 && w.style.color !== '#e0e0e0') {
                        w.style.color = '#e0e0e0';
                        w.style.fontWeight = '600';
                    }
                });
            });
            animationId = requestAnimationFrame(updateUI);
        }

        playBtn.onclick = () => {
            if (audioPlayer.paused) {
                audioPlayer.play();
                playBtn.textContent = '⏸';
            } else {
                audioPlayer.pause();
                playBtn.textContent = '▶';
            }
        };

        seekSlider.oninput = (e) => {
            if (!audioPlayer || !audioPlayer.duration) return;
            const val = parseFloat(e.target.value);
            audioPlayer.currentTime = (val / 100) * audioPlayer.duration;
        };

        audioPlayer.onended = () => { playBtn.textContent = '▶'; };
        audioPlayer.onloadedmetadata = () => {
            timeDisplay.textContent = `00:00 / ${formatTime(audioPlayer.duration)}`;
        };

        updateUI();
    }

    // =================================================================
    // ===== ЛОГИКА ВКЛАДОК И НОВЫЙ РЕДАКТОР =====
    // =================================================================
    
    // Переключение вкладок
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            
            // Сбрасываем активные классы
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            // Активируем нужную вкладку
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
            
            // Если открыли редактор, а он пустой — рендерим
            if (btn.dataset.target === 'editorTab' && editorLines.length === 0 && window.lastGeneratedLines) {
                openEditor(window.lastGeneratedLines, fileToProcess);
            }
        });
    });

    let editorLines = [];
    let editorAudio = null;
    let editorAudioUrl = null;
    let editorPlayTimeout = null;
    let editorAnimFrame = null;
    let waveformPeaks = [];
    let selectedWord = null;
    let undoStack = [];
    let redoStack = [];

    const cloneLines = (lines) => JSON.parse(JSON.stringify(lines || []));
    const roundMs = (value) => Math.max(0, Math.round(Number(value || 0) * 1000) / 1000);

    function metadataLines() {
        const keys = ['ar', 'ti', 'al', 'by', 're', 've'];
        const values = { ...currentMetadata, ar: artistInput.value.trim(), ti: titleInput.value.trim(), al: albumInput.value.trim() };
        return keys.filter(key => values[key]).map(key => `[${key}:${String(values[key]).replace(/[\r\n]/g, ' ')}]`);
    }

    function editorLineEnd(line, lineIdx) {
        const lastWord = line.words && line.words[line.words.length - 1];
        if (lastWord && Number.isFinite(Number(lastWord.end))) return Number(lastWord.end);
        if (Number.isFinite(Number(line.end))) return Number(line.end);
        if (editorLines[lineIdx + 1]) return editorLines[lineIdx + 1].start;
        return (editorAudio && editorAudio.duration) || Number(line.start) + 4;
    }

    function generateElrcFromLines(compatible = false) {
        const precision = compatible ? 2 : 3;
        const stamp = (seconds) => {
            const precise = formatTimestamp(seconds);
            return precision === 2 ? precise.slice(0, -1) : precise;
        };
        const body = editorLines.map((line, lineIdx) => {
            const lineTag = `[${stamp(line.start)}]`;
            if (!line.words || !line.words.length) return lineTag + (line.text || '');
            const words = compatible
                ? line.words.map(word => `<${stamp(word.start)}>${String(word.word || '').trim()}`).join(' ')
                : line.words.map(word => `<${stamp(word.start)}>${String(word.word || '').trim()} <${stamp(word.end)}>`).join(' ');
            return lineTag + words;
        });
        const headers = metadataLines();
        return headers.length ? [...headers, '', ...body].join('\n') : body.join('\n');
    }

    function generateSimpleLrcFromLines() {
        const body = editorLines.map(line => `[${formatTimestamp(line.start)}]${line.words?.length ? line.words.map(w => w.word).join(' ') : (line.text || '')}`);
        const headers = metadataLines();
        return headers.length ? [...headers, '', ...body].join('\n') : body.join('\n');
    }

    function refreshEditorOutputs() {
        currentOutputs = {
            elrc: generateElrcFromLines(false),
            elrcCompatible: generateElrcFromLines(true),
            lrc: generateSimpleLrcFromLines()
        };
        currentLrc = selectOutputByFormat();
        updateLrcPreview();
    }

    function updateLrcPreview() {
        const preview = document.getElementById('editorLrcPreview');
        if (preview) preview.textContent = generateElrcFromLines(false);
    }

    function recordHistory() {
        undoStack.push(cloneLines(editorLines));
        if (undoStack.length > 100) undoStack.shift();
        redoStack = [];
        updateHistoryButtons();
    }

    function updateHistoryButtons() {
        document.getElementById('undoEditorBtn').disabled = undoStack.length === 0;
        document.getElementById('redoEditorBtn').disabled = redoStack.length === 0;
    }

    function afterEditorMutation(fullRender = true) {
        editorLines.forEach(line => {
            line.text = (line.words || []).map(word => word.word).join(' ').trim() || line.text || '';
        });
        if (fullRender) renderEditor();
        buildEditorKaraoke();
        refreshEditorOutputs();
        validateEditor(false);
    }

    function mutateEditor(callback, fullRender = true) {
        recordHistory();
        callback();
        afterEditorMutation(fullRender);
    }

    function seedMissingWordTimings(lines) {
        lines.forEach((line, lineIdx) => {
            if (line.words?.length || !String(line.text || '').trim()) return;
            const tokens = String(line.text).trim().split(/\s+/);
            const nextStart = lines[lineIdx + 1]?.start;
            const end = Number(line.end) > Number(line.start)
                ? Number(line.end)
                : (Number(nextStart) > Number(line.start) ? Number(nextStart) : Number(line.start) + Math.max(1, tokens.length * .35));
            const step = (end - Number(line.start)) / tokens.length;
            line.words = tokens.map((word, wordIdx) => ({
                word,
                start: roundMs(Number(line.start) + step * wordIdx),
                end: roundMs(Number(line.start) + step * (wordIdx + 1))
            }));
            line.end = roundMs(end);
        });
        return lines;
    }

    function openEditor(lines, audioFile) {
        editorLines = seedMissingWordTimings(cloneLines(lines));
        selectedWord = null;
        undoStack = [];
        redoStack = [];
        if (editorAudio) editorAudio.pause();
        if (editorAudioUrl) URL.revokeObjectURL(editorAudioUrl);
        editorAudioUrl = audioFile ? URL.createObjectURL(audioFile) : null;
        editorAudio = editorAudioUrl ? new Audio(editorAudioUrl) : null;
        if (editorAudio) editorAudio.playbackRate = parseFloat(document.getElementById('editorSpeedSelect').value) || 0.5;
        renderEditor();
        buildEditorKaraoke();
        refreshEditorOutputs();
        setupEditorPlayer();
        updateHistoryButtons();
        validateEditor(false);
        if (audioFile) buildWaveform(audioFile);
    }

    async function buildWaveform(audioFile) {
        const canvas = document.getElementById('editorWaveform');
        try {
            const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
            const context = new AudioContextCtor();
            const buffer = await context.decodeAudioData(await audioFile.arrayBuffer());
            const samples = buffer.getChannelData(0);
            const count = Math.max(300, Math.floor(canvas.clientWidth * (window.devicePixelRatio || 1)));
            const block = Math.max(1, Math.floor(samples.length / count));
            waveformPeaks = [];
            for (let i = 0; i < count; i++) {
                let peak = 0;
                const end = Math.min(samples.length, (i + 1) * block);
                for (let j = i * block; j < end; j++) peak = Math.max(peak, Math.abs(samples[j]));
                waveformPeaks.push(peak);
            }
            await context.close();
            drawWaveform();
        } catch (_) {
            waveformPeaks = [];
            drawWaveform();
        }
    }

    function drawWaveform() {
        const canvas = document.getElementById('editorWaveform');
        const ratio = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
        const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = 'rgba(3,0,20,.9)';
        ctx.fillRect(0, 0, width, height);
        if (waveformPeaks.length) {
            ctx.strokeStyle = 'rgba(168,85,247,.75)';
            ctx.lineWidth = Math.max(1, ratio);
            ctx.beginPath();
            waveformPeaks.forEach((peak, index) => {
                const x = index / Math.max(1, waveformPeaks.length - 1) * width;
                const amplitude = peak * height * .46;
                ctx.moveTo(x, height / 2 - amplitude);
                ctx.lineTo(x, height / 2 + amplitude);
            });
            ctx.stroke();
        }
        if (editorAudio && editorAudio.duration) {
            const x = editorAudio.currentTime / editorAudio.duration * width;
            ctx.strokeStyle = '#f0eef6';
            ctx.lineWidth = 2 * ratio;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        }
    }

    function buildEditorKaraoke() {
        const container = document.getElementById('editorKaraoke');
        container.innerHTML = '';
        editorLines.forEach((line, lineIdx) => {
            const div = document.createElement('div');
            div.className = 'ek-line';
            div.dataset.index = lineIdx;
            div.dataset.start = Number(line.start) || 0;
            (line.words || []).forEach((word, wordIdx) => {
                const span = document.createElement('span');
                span.className = 'ek-word idle';
                span.textContent = word.word + ' ';
                span.dataset.start = Number(word.start) || 0;
                span.dataset.end = Number(word.end) || 0;
                span.onclick = () => { selectedWord = { lineIdx, wordIdx }; renderEditor(); };
                div.appendChild(span);
            });
            if (!line.words?.length) div.textContent = line.text || '';
            container.appendChild(div);
        });
    }

    function updateEditorPlaybackUI() {
        if (!editorAudio) return;
        const cur = editorAudio.currentTime;
        const dur = editorAudio.duration || 0;
        document.getElementById('editorTimeDisplay').textContent = `${formatTimestamp(cur)} / ${formatTimestamp(dur)}`;
        document.getElementById('editorSeekSlider').value = dur ? (cur / dur) * 100 : 0;
        document.querySelectorAll('#editorKaraoke .ek-line').forEach((el, idx, all) => {
            const start = Number(el.dataset.start);
            const end = all[idx + 1] ? Number(all[idx + 1].dataset.start) : Infinity;
            const active = cur >= start && cur < end;
            el.classList.toggle('active', active);
            if (active) el.scrollIntoView({ block: 'nearest' });
            el.querySelectorAll('.ek-word').forEach(word => {
                const ws = Number(word.dataset.start);
                const we = Number(word.dataset.end);
                word.className = cur >= we ? 'ek-word done' : cur >= ws ? 'ek-word current' : 'ek-word idle';
            });
        });
        drawWaveform();
        if (!editorAudio.paused) editorAnimFrame = requestAnimationFrame(updateEditorPlaybackUI);
    }

    function setupEditorPlayer() {
        const playBtn = document.getElementById('editorPlayBtn');
        const seekSlider = document.getElementById('editorSeekSlider');
        const canvas = document.getElementById('editorWaveform');
        if (!editorAudio) return;
        playBtn.onclick = () => toggleEditorPlayback();
        seekSlider.oninput = event => {
            if (editorAudio.duration) editorAudio.currentTime = Number(event.target.value) / 100 * editorAudio.duration;
            updateEditorPlaybackUI();
        };
        canvas.onclick = event => {
            if (!editorAudio.duration) return;
            const rect = canvas.getBoundingClientRect();
            editorAudio.currentTime = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * editorAudio.duration;
            updateEditorPlaybackUI();
        };
        editorAudio.onloadedmetadata = updateEditorPlaybackUI;
        editorAudio.onended = () => { playBtn.textContent = '▶'; updateEditorPlaybackUI(); };
    }

    function toggleEditorPlayback() {
        if (!editorAudio) return;
        const button = document.getElementById('editorPlayBtn');
        if (editorAudio.paused) {
            editorAudio.play();
            button.textContent = '⏸';
            if (editorAnimFrame) cancelAnimationFrame(editorAnimFrame);
            editorAnimFrame = requestAnimationFrame(updateEditorPlaybackUI);
        } else {
            editorAudio.pause();
            button.textContent = '▶';
            if (editorAnimFrame) cancelAnimationFrame(editorAnimFrame);
            updateEditorPlaybackUI();
        }
    }

    function button(label, title, handler, className = 'word-btn') {
        const element = document.createElement('button');
        element.type = 'button'; element.className = className; element.textContent = label; element.title = title;
        element.onclick = event => { event.stopPropagation(); handler(); };
        return element;
    }

    function renderEditor() {
        const container = document.getElementById('editorLines');
        container.innerHTML = '';
        editorLines.forEach((line, lineIdx) => {
            const row = document.createElement('div');
            row.className = 'editor-line-card' + (selectedWord?.lineIdx === lineIdx ? ' selected-line' : '');
            const header = document.createElement('div'); header.className = 'editor-line-header';
            const number = document.createElement('span'); number.className = 'line-number'; number.textContent = `#${lineIdx + 1}`;
            const lineTime = document.createElement('input');
            lineTime.className = 'editor-input time-input'; lineTime.value = formatTimestamp(line.start); lineTime.title = 'Начало строки';
            lineTime.onchange = event => {
                const value = parseTimestamp(event.target.value);
                if (Number.isFinite(value)) mutateEditor(() => { line.start = roundMs(value); }, false);
                else event.target.value = formatTimestamp(line.start);
            };
            const actions = document.createElement('div'); actions.className = 'editor-line-actions';
            actions.append(
                button('▶', 'Прослушать строку', () => playRange(line.start, editorLineEnd(line, lineIdx))),
                button('◎', 'Начало строки = текущая позиция', () => mutateEditor(() => { line.start = roundMs(editorAudio?.currentTime); })),
                button('＋ слово', 'Добавить слово', () => addWord(lineIdx)),
                button('Разделить', 'Разделить перед выбранным словом', () => splitLine(lineIdx)),
                button('✕', 'Удалить строку', () => { if (confirm(`Удалить строку ${lineIdx + 1}?`)) mutateEditor(() => editorLines.splice(lineIdx, 1)); }, 'word-btn del-btn')
            );
            header.append(number, lineTime, actions); row.appendChild(header);

            const words = document.createElement('div'); words.className = 'editor-words-grid';
            (line.words || []).forEach((word, wordIdx) => {
                const card = document.createElement('div');
                card.className = 'word-card' + (selectedWord?.lineIdx === lineIdx && selectedWord?.wordIdx === wordIdx ? ' selected-word' : '');
                card.onclick = event => {
                    if (event.target !== card) return;
                    selectedWord = { lineIdx, wordIdx };
                    renderEditor();
                };
                const text = document.createElement('input'); text.className = 'editor-input word-text'; text.value = word.word || '';
                text.onfocus = () => { selectedWord = { lineIdx, wordIdx }; card.classList.add('selected-word'); };
                text.onchange = event => mutateEditor(() => { word.word = event.target.value.trim(); });
                const timings = document.createElement('div'); timings.className = 'word-timings';
                const start = document.createElement('input'); start.className = 'editor-input time-input micro'; start.value = formatTimestamp(word.start);
                const end = document.createElement('input'); end.className = 'editor-input time-input micro'; end.value = formatTimestamp(word.end);
                start.onfocus = end.onfocus = () => { selectedWord = { lineIdx, wordIdx }; card.classList.add('selected-word'); };
                start.onchange = event => changeWordTime(word, 'start', event.target.value, start);
                end.onchange = event => changeWordTime(word, 'end', event.target.value, end);
                timings.append(start, document.createTextNode('—'), end);
                const controls = document.createElement('div'); controls.className = 'word-controls';
                controls.append(
                    button('▶', 'Прослушать слово', () => { selectedWord = { lineIdx, wordIdx }; playRange(word.start, word.end, card); }),
                    button('S', 'Поставить начало по текущей позиции', () => { selectedWord = { lineIdx, wordIdx }; setSelectedBoundary('start'); }, 'word-btn capture-btn'),
                    button('E', 'Поставить конец по текущей позиции', () => { selectedWord = { lineIdx, wordIdx }; setSelectedBoundary('end'); }, 'word-btn capture-btn'),
                    button('✕', 'Удалить слово', () => mutateEditor(() => { line.words.splice(wordIdx, 1); selectedWord = null; }), 'word-btn del-btn')
                );
                card.append(text, timings, controls); words.appendChild(card);
            });
            if (!line.words?.length) words.textContent = line.text || 'В строке нет слов';
            row.appendChild(words); container.appendChild(row);
        });
    }

    function changeWordTime(word, field, raw, input) {
        const value = parseTimestamp(raw);
        if (!Number.isFinite(value)) { input.value = formatTimestamp(word[field]); return; }
        mutateEditor(() => { word[field] = roundMs(value); }, false);
    }

    function playRange(start, end, card = null) {
        if (!editorAudio) return;
        const from = Math.max(0, Number(start) || 0);
        const to = Math.max(from + .01, Number(end) || from + 1);
        if (editorPlayTimeout) clearTimeout(editorPlayTimeout);
        editorAudio.pause(); editorAudio.currentTime = from; editorAudio.play();
        document.getElementById('editorPlayBtn').textContent = '⏸';
        if (card) card.classList.add('playing');
        editorAnimFrame = requestAnimationFrame(updateEditorPlaybackUI);
        editorPlayTimeout = setTimeout(() => {
            editorAudio.pause(); if (card) card.classList.remove('playing');
            document.getElementById('editorPlayBtn').textContent = '▶';
        }, (to - from) * 1000 / editorAudio.playbackRate + 25);
    }

    function addWord(lineIdx) {
        const line = editorLines[lineIdx];
        mutateEditor(() => {
            const last = line.words?.[line.words.length - 1];
            const start = last ? Number(last.end) : Number(line.start);
            line.words = line.words || [];
            line.words.push({ word: 'слово', start: roundMs(start), end: roundMs(start + .4) });
            selectedWord = { lineIdx, wordIdx: line.words.length - 1 };
        });
    }

    function splitLine(lineIdx) {
        if (selectedWord?.lineIdx !== lineIdx || selectedWord.wordIdx <= 0) {
            showToast('Выбери слово, перед которым разделить строку', 'error'); return;
        }
        mutateEditor(() => {
            const line = editorLines[lineIdx];
            const tail = line.words.splice(selectedWord.wordIdx);
            editorLines.splice(lineIdx + 1, 0, { start: tail[0].start, end: line.end, text: tail.map(w => w.word).join(' '), words: tail });
            line.end = Math.max(line.start + .01, tail[0].start);
            selectedWord = { lineIdx: lineIdx + 1, wordIdx: 0 };
        });
    }

    function setSelectedBoundary(field) {
        if (!selectedWord || !editorAudio) { showToast('Сначала выбери слово', 'error'); return; }
        const { lineIdx, wordIdx } = selectedWord;
        const line = editorLines[lineIdx]; const word = line?.words?.[wordIdx];
        if (!word) return;
        mutateEditor(() => {
            const now = roundMs(editorAudio.currentTime);
            if (field === 'start') {
                word.start = now;
                if (!(Number(word.end) > now)) word.end = roundMs(now + .05);
                if (wordIdx === 0) line.start = word.start;
                const previous = line.words[wordIdx - 1];
                if (previous && previous.end > now) previous.end = now;
            } else {
                word.end = Math.max(now, Number(word.start) + .001);
                const next = line.words[wordIdx + 1];
                if (next && next.start < word.end) next.start = word.end;
                if (wordIdx === line.words.length - 1) line.end = word.end;
            }
        });
    }

    function validationIssues() {
        const issues = [];
        editorLines.forEach((line, lineIdx) => {
            if (lineIdx && Number(line.start) < Number(editorLines[lineIdx - 1].start)) issues.push(`Строка ${lineIdx + 1}: нарушен порядок`);
            if (!line.words?.length) issues.push(`Строка ${lineIdx + 1}: нет пословной разметки`);
            (line.words || []).forEach((word, wordIdx) => {
                if (!String(word.word || '').trim()) issues.push(`Строка ${lineIdx + 1}, слово ${wordIdx + 1}: пустой текст`);
                if (Number(word.start) < Number(line.start)) issues.push(`Строка ${lineIdx + 1}, слово ${wordIdx + 1}: раньше строки`);
                if (!(Number(word.end) > Number(word.start))) issues.push(`Строка ${lineIdx + 1}, слово ${wordIdx + 1}: неверный интервал`);
                const next = line.words[wordIdx + 1];
                if (next && Number(next.start) < Number(word.start)) issues.push(`Строка ${lineIdx + 1}: нарушен порядок слов`);
                if (next && Number(word.end) > Number(next.start)) issues.push(`Строка ${lineIdx + 1}: слова ${wordIdx + 1}–${wordIdx + 2} пересекаются`);
            });
        });
        return issues;
    }

    function repairEditorTimings() {
        let previousLine = 0;
        editorLines.forEach((line, lineIdx) => {
            line.start = roundMs(Math.max(lineIdx ? previousLine : 0, Number(line.start) || 0));
            previousLine = line.start;
            line.words = (line.words || []).filter(word => String(word.word || '').trim());
            let cursor = line.start;
            line.words.forEach((word, wordIdx) => {
                word.start = roundMs(Math.max(cursor, Number(word.start) || cursor));
                const nextStart = line.words[wordIdx + 1] ? Number(line.words[wordIdx + 1].start) : Infinity;
                word.end = roundMs(Math.max(word.start + .001, Math.min(Number(word.end) || word.start + .3, nextStart)));
                cursor = word.start;
            });
            if (line.words.length) { line.start = Math.min(line.start, line.words[0].start); line.end = line.words[line.words.length - 1].end; }
        });
    }

    function validateEditor(showResult = true) {
        const status = document.getElementById('editorStatus');
        const issues = validationIssues();
        status.className = 'editor-status ' + (issues.length ? 'warn' : 'ok');
        status.textContent = issues.length ? `${issues.length} замечаний. Нажми «Проверить», чтобы исправить автоматически.` : `Тайминги корректны: ${editorLines.length} строк, ${editorLines.reduce((n, l) => n + (l.words?.length || 0), 0)} слов`;
        if (showResult && issues.length) {
            mutateEditor(repairEditorTimings);
            const remaining = validationIssues().length;
            showToast(remaining ? `Осталось замечаний: ${remaining}` : `Исправлено замечаний: ${issues.length}`, remaining ? 'error' : 'success');
        } else if (showResult) showToast('Ошибок в таймингах нет', 'success');
    }

    document.getElementById('stopEditorBtn').addEventListener('click', () => {
        if (editorAudio) editorAudio.pause();
        if (editorPlayTimeout) clearTimeout(editorPlayTimeout);
        if (editorAnimFrame) cancelAnimationFrame(editorAnimFrame);
        document.getElementById('editorPlayBtn').textContent = '▶';
        document.querySelectorAll('.word-card').forEach(card => card.classList.remove('playing'));
        drawWaveform();
    });
    document.getElementById('editorSpeedSelect').addEventListener('change', function() { if (editorAudio) editorAudio.playbackRate = Number(this.value); });
    document.getElementById('validateEditorBtn').addEventListener('click', () => validateEditor(true));
    document.getElementById('undoEditorBtn').addEventListener('click', () => {
        if (!undoStack.length) return; redoStack.push(cloneLines(editorLines)); editorLines = undoStack.pop(); selectedWord = null; afterEditorMutation(); updateHistoryButtons();
    });
    document.getElementById('redoEditorBtn').addEventListener('click', () => {
        if (!redoStack.length) return; undoStack.push(cloneLines(editorLines)); editorLines = redoStack.pop(); selectedWord = null; afterEditorMutation(); updateHistoryButtons();
    });
    document.getElementById('applyOffsetBtn').addEventListener('click', () => {
        const seconds = Number(document.getElementById('editorOffsetInput').value) / 1000;
        if (!Number.isFinite(seconds) || seconds === 0) return;
        mutateEditor(() => editorLines.forEach(line => {
            line.start = roundMs(line.start + seconds); line.end = roundMs((line.end || line.start) + seconds);
            (line.words || []).forEach(word => { word.start = roundMs(word.start + seconds); word.end = roundMs(word.end + seconds); });
        }));
        document.getElementById('editorOffsetInput').value = 0;
    });
    document.getElementById('addLineBtn').addEventListener('click', () => {
        mutateEditor(() => {
            const last = editorLines[editorLines.length - 1];
            const start = last ? editorLineEnd(last, editorLines.length - 1) + .25 : (editorAudio?.currentTime || 0);
            editorLines.push({ start: roundMs(start), end: roundMs(start + 1), text: '', words: [] });
        });
    });

    document.addEventListener('keydown', event => {
        const editable = /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); document.getElementById(event.shiftKey ? 'redoEditorBtn' : 'undoEditorBtn').click(); return; }
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') { event.preventDefault(); document.getElementById('redoEditorBtn').click(); return; }
        if (editable || !document.getElementById('editorTab').classList.contains('active')) return;
        if (event.code === 'Space') { event.preventDefault(); toggleEditorPlayback(); }
        else if (event.key.toLowerCase() === 's') { event.preventDefault(); setSelectedBoundary('start'); }
        else if (event.key.toLowerCase() === 'e') { event.preventDefault(); setSelectedBoundary('end'); }
        else if (selectedWord && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
            event.preventDefault(); const delta = (event.shiftKey ? .1 : .01) * (event.key === 'ArrowLeft' ? -1 : 1);
            const word = editorLines[selectedWord.lineIdx]?.words?.[selectedWord.wordIdx];
            if (word) mutateEditor(() => { word.start = roundMs(word.start + delta); word.end = roundMs(word.end + delta); });
        }
    });

    window.addEventListener('resize', drawWaveform);

    document.getElementById('saveEditorBtn').addEventListener('click', () => {
        if (validationIssues().length) { showToast('Сначала исправь ошибки кнопкой «Проверить»', 'error'); return; }
        refreshEditorOutputs();
        lrcOutput.textContent = currentLrc;
        window.lastGeneratedLines = cloneLines(editorLines);
        if (fileToProcess) setupPlayer(editorLines, fileToProcess);
        showToast('Изменения применены. ELRC готов.', 'success');
        document.querySelector('[data-target="generatorTab"]').click();
    });

    // ===== ВСПОМОГАТЕЛЬНЫЕ =====
    function addLog(msg) {
        const entry = document.createElement('div');
        entry.textContent = msg;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    function updateProgress(value, message) {
        progressFill.style.width = value + '%';
        progressPercent.textContent = Math.round(value) + '%';
        progressLabel.textContent = message;
        if (value >= 10 && value < 30) setStepActive(step1);
        else if (value >= 30 && value < 70) { setStepDone(step1); setStepActive(step2); }
        else if (value >= 70 && value < 95) { setStepDone(step2); setStepActive(step3); }
        else if (value >= 95) setStepDone(step3);
    }

    function setStepActive(el) { el.classList.remove('done'); el.classList.add('active'); }
    function setStepDone(el) { el.classList.remove('active'); el.classList.add('done'); }
    function resetUI() {
        generateBtn.disabled = false;
        generateBtn.classList.remove('loading');
        generateBtn.querySelector('.btn-text').textContent = 'Создать ELRC';
    }

    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(currentLrc).then(() => {
            copyBtn.textContent = '✅ Скопировано!';
            setTimeout(() => copyBtn.textContent = '📋 Копировать', 2000);
        });
    });

    editBtn.addEventListener('click', () => {
        if (!window.lastGeneratedLines) return;
        document.querySelector('[data-target="editorTab"]').click();
    });

    function downloadText(text, filename) {
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    downloadBtn.addEventListener('click', () => downloadText(
        selectOutputByFormat(),
        formatSelect.value === 'lrc' ? (currentLrcFilename || 'lyrics.lrc') : (currentFilename || 'lyrics.elrc')
    ));
    downloadSimpleBtn.addEventListener('click', () => downloadText(currentOutputs.lrc, currentLrcFilename || 'lyrics.lrc'));

    function showToast(msg, type = 'error') {
        const icon = type === 'error' ? '❌' : '✅';
        toast.className = `toast ${type}`;
        toast.innerHTML = icon + ' ' + msg;
        toast.classList.add('visible');
        setTimeout(() => toast.classList.remove('visible'), 3500);
    }

    // ===== СКАЧИВАНИЕ (track-dl) =====
    const tgStatus = document.getElementById('tgStatus');
    const tgQuery = document.getElementById('tgQuery');
    const tgSearchBtn = document.getElementById('tgSearchBtn');
    const tgResult = document.getElementById('tgResult');

    function tgRunGenerate(track) {
        tgSearchBtn.disabled = true;
        tgSearchBtn.textContent = 'Обработка…';
        progressFill.style.width = '0%';
        progressLabel.textContent = 'Скачивание…';
        progressPercent.textContent = '0%';
        [step1, step2, step3].forEach(s => s.classList.remove('active', 'done'));
        logContainer.innerHTML = '';
        logContainer.style.display = 'block';
        addLog('🎵 Скачиваем трек…');
        addLog(`🎤 ${[track.artist, track.title].filter(Boolean).join(' — ') || track.query}`);
        resultSection.classList.remove('visible');
        playerContainer.classList.add('hidden');
        progressSection.classList.add('visible');

        fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file: track.file,
                artist: track.artist,
                title: track.title,
                album: track.album,
                language: languageSelect.value,
                use_demucs: true
            })
        })
        .then(async res => {
            let resultData = null;
            await consumeSSEResponse(res, data => {
                if (data.type === 'progress') {
                    updateProgress(data.percent, data.message);
                    addLog(`▸ ${data.message} (${data.percent}%)`);
                } else if (data.type === 'result') resultData = data;
                else if (data.type === 'error') throw new Error(data.message);
            });
            if (resultData) {
                fetch('/downloads/' + encodeURIComponent(resultData.mp3))
                    .then(r => r.blob())
                    .then(blob => {
                        const audioFile = new File([blob], resultData.mp3, { type: 'audio/mpeg' });
                        applyResult(resultData, audioFile);
                    })
                    .catch(() => applyResult(resultData, null));
            } else throw new Error('Сервер не вернул результат');
        })
        .catch(err => {
            showToast('Ошибка: ' + err.message, 'error');
            resetUI();
            addLog('❌ Ошибка: ' + err.message);
        })
        .finally(() => {
            tgSearchBtn.disabled = false;
            tgSearchBtn.textContent = 'Скачать 🎵';
        });
    }

    tgSearchBtn.addEventListener('click', () => {
        const query = tgQuery.value.trim();
        if (!query) return showToast('Введи название песни', 'error');
        tgSearchBtn.disabled = true;
        tgSearchBtn.textContent = 'Скачиваю…';
        tgStatus.textContent = 'скачивание с YouTube…';
        tgResult.innerHTML = '';
        fetch('/trackdl', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                tgResult.innerHTML = `<div class="tg-error">${String(d.error).replace(/\n/g, '<br>')}</div>`;
                tgStatus.textContent = 'ошибка';
                return;
            }
            tgStatus.textContent = 'готово';
            tgRunGenerate(d);
        })
        .catch(e => {
            tgResult.innerHTML = `<div class="tg-error">${String(e.message).replace(/\n/g, '<br>')}</div>`;
            tgStatus.textContent = 'ошибка';
        })
        .finally(() => {
            if (tgSearchBtn.textContent === 'Скачиваю…') {
                tgSearchBtn.disabled = false;
                tgSearchBtn.textContent = 'Скачать 🎵';
            }
        });
    });

    tgQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') tgSearchBtn.click();
    });

    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
})();
