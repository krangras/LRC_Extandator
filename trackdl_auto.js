#!/usr/bin/env node
// Автоматическая обёртка над track-dl: без интерактивных подсказок.
// Берёт первый результат YouTube, первую метадату и обложку (Deezer/iTunes),
// максимальный битрейт источника, таргет 192 kbps.
// Результат пишет в trackdl_result.json в рабочей папке (cwd).

const path = require('path');
const fs = require('fs');
const { searchYouTube, downloadYouTubeAudioWithTemp, getAudioFormats } = require('track-dl/lib/youtube');
const { mergeMetadata } = require('track-dl/lib/merger');
const { fetchSongInfoOptions, fetchCoverOptions, parseYouTubeTitle } = require('track-dl/lib/metadata');

const LIMIT = 6;
const RESULT_FILE = 'trackdl_result.json';

function sanitizeFilename(name) {
  return (name || '')
    .replace(/[<>:"/\\|?*$`]/g, '')
    .replace(/[\u0000-\u001f\u007f]/g, '')
    .replace(/[. ]+$/g, '')
    .trim();
}

function done(data) {
  fs.writeFileSync(RESULT_FILE, JSON.stringify(data, null, 2), 'utf-8');
  process.exit(0);
}

function fail(message) {
  done({ error: message });
}

async function run() {
  const query = process.argv.slice(2).join(' ').trim();
  if (!query) return fail('Не указан запрос');

  const youtubeResults = await searchYouTube(query, LIMIT);
  if (!youtubeResults.length) {
    return fail('YouTube не вернул результатов. Возможно, YouTube недоступен из твоей сети.');
  }

  const video = youtubeResults[0];
  const parsed = parseYouTubeTitle(video.title);
  const metaQuery = [parsed.artist, parsed.title].filter(Boolean).join(' ') || query;

  let metadata = null;
  try {
    const options = await fetchSongInfoOptions(metaQuery, LIMIT);
    if (options.length) metadata = options[0];
  } catch (e) {
    console.error('metadata error:', e.message);
  }

  let cover = null;
  if (metadata) {
    try {
      const options = await fetchCoverOptions(metadata, LIMIT);
      if (options.length) cover = options[0];
    } catch (e) {
      console.error('cover error:', e.message);
    }
  }

  let sourceFormatId = null;
  try {
    const formats = await getAudioFormats(video.url);
    if (formats.length) sourceFormatId = formats[formats.length - 1].format_id;
  } catch (e) {
    console.error('formats error:', e.message);
  }

  const artist = (metadata && metadata.artist) || parsed.artist || video.uploader || '';
  const title = (metadata && metadata.title) || parsed.title || video.title || '';
  const album = (metadata && metadata.album) || '';

  const tempAudioPath = await downloadYouTubeAudioWithTemp(video.url, sourceFormatId);
  if (!tempAudioPath) return fail('Не удалось скачать аудио с YouTube.');

  let name = [artist, title].filter(Boolean).join(' - ');
  if (!name.trim()) name = sanitizeFilename(video.title) || 'track';
  const outputPath = path.join(process.cwd(), `${sanitizeFilename(name)}.mp3`);

  await mergeMetadata(tempAudioPath, {
    title: title || video.title,
    artist: artist || video.uploader || '',
    album: album,
    year: (metadata && metadata.year) || '',
    genre: (metadata && metadata.genre) || '',
    albumArt: (cover && cover.url) || ''
  }, outputPath, 192);

  done({
    file: path.basename(outputPath),
    artist: artist,
    title: title,
    album: album,
    query: query,
    cover: (cover && cover.url) || ''
  });
}

run().catch((e) => fail(e.message || String(e)));
