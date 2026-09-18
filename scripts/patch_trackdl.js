#!/usr/bin/env node
// track-dl жёстко прописывает путь к yt-dlp.exe (Windows).
// На Linux/macOS подменяем его на системный yt-dlp.
// Запускается автоматически после npm install (см. install.sh).

const fs = require('fs');
const path = require('path');

const file = path.join(__dirname, '..', 'node_modules', 'track-dl', 'lib', 'youtube.js');
if (!fs.existsSync(file)) {
  console.error('patch_trackdl: youtube.js не найден — сначала npm install');
  process.exit(1);
}

let src = fs.readFileSync(file, 'utf-8');
const original = "const YTDLP_PATH = path.join(__dirname, '..', 'yt-dlp.exe');";
const patched = `const YTDLP_PATH = process.platform === 'win32'
  ? path.join(__dirname, '..', 'yt-dlp.exe')
  : (process.env.YTDLP_PATH || 'yt-dlp');`;

if (src.includes(patched)) {
  console.log('patch_trackdl: уже пропатчен');
} else if (src.includes(original)) {
  fs.writeFileSync(file, src.replace(original, patched), 'utf-8');
  console.log('patch_trackdl: youtube.js пропатчен для Linux/macOS');
} else {
  console.error('patch_trackdl: не найден ожидаемый код — версия track-dl изменилась?');
  process.exit(1);
}
