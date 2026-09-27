(() => {
  'use strict';

  const SETTINGS_KEY = 'lrcExtandatorV6Settings';
  const defaults = { qualityMode: 'max', demucsMode: 'auto' };
  let settings = { ...defaults };
  try {
    settings = { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch {}

  const save = () => localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));

  function addSettings() {
    const host = document.querySelector('.settings-row');
    if (!host || document.getElementById('v6QualityMode')) return;

    const quality = document.createElement('div');
    quality.className = 'setting-item v6-setting';
    quality.innerHTML = `
      <label for="v6QualityMode">Качество синхронизации</label>
      <div class="select-wrapper">
        <select id="v6QualityMode">
          <option value="max">Максимум качества</option>
          <option value="balanced">Баланс</option>
          <option value="fast">Быстро</option>
        </select>
      </div>`;

    const demucs = document.createElement('div');
    demucs.className = 'setting-item v6-setting';
    demucs.innerHTML = `
      <label for="v6DemucsMode">Вокальный stem</label>
      <div class="select-wrapper">
        <select id="v6DemucsMode">
          <option value="auto">Авто — только если помогает</option>
          <option value="true">Всегда Demucs</option>
          <option value="false">Без Demucs</option>
        </select>
      </div>`;

    host.append(quality, demucs);
    const q = document.getElementById('v6QualityMode');
    const d = document.getElementById('v6DemucsMode');
    q.value = settings.qualityMode;
    d.value = settings.demucsMode;
    q.addEventListener('change', () => { settings.qualityMode = q.value; save(); });
    d.addEventListener('change', () => { settings.demucsMode = d.value; save(); });
  }

  function ensureQualityCard() {
    const result = document.getElementById('resultSection');
    if (!result) return null;
    let card = document.getElementById('v6QualityCard');
    if (card) return card;
    card = document.createElement('div');
    card.id = 'v6QualityCard';
    card.className = 'v6-quality-card';
    card.innerHTML = `
      <div class="v6-quality-head">
        <strong>Alignment Quality</strong>
        <span id="v6QualityGrade">—</span>
      </div>
      <div class="v6-quality-grid">
        <div><span>Score</span><b id="v6QualityScore">—</b></div>
        <div><span>Aligned</span><b id="v6QualityAligned">—</b></div>
        <div><span>Interpolated</span><b id="v6QualityInterpolated">—</b></div>
        <div><span>Anchor MAE</span><b id="v6QualityAnchor">—</b></div>
      </div>
      <div id="v6QualityNote" class="v6-quality-note"></div>`;
    result.prepend(card);
    return card;
  }

  function renderQuality(resultData) {
    const q = resultData?.quality;
    if (!q) return;
    const card = ensureQualityCard();
    if (!card) return;
    const grade = String(q.grade || 'unknown');
    card.dataset.grade = grade;
    document.getElementById('v6QualityGrade').textContent = grade.toUpperCase();
    document.getElementById('v6QualityScore').textContent = Number(q.score || 0).toFixed(3);
    document.getElementById('v6QualityAligned').textContent = `${Number(q.percent?.aligned ?? ((q.alignedWordRatio ?? 0) * 100)).toFixed(1)}%`;
    document.getElementById('v6QualityInterpolated').textContent = `${Number(q.percent?.interpolated ?? ((q.interpolatedWordRatio ?? 0) * 100)).toFixed(1)}%`;
    document.getElementById('v6QualityAnchor').textContent = q.anchorMaeMs == null ? '—' : `${Math.round(q.anchorMaeMs)} ms`;
    const selected = resultData.alignment?.selectedCandidate;
    const cache = resultData.alignment?.cacheHit ? ' · cache' : '';
    const candidate = selected ? ` · ${selected.label || `${selected.use_demucs ? 'vocals' : 'mix'}/${selected.repetition}`}` : '';
    document.getElementById('v6QualityNote').textContent = q.publishable
      ? `✓ Результат прошёл автоматический quality gate${candidate}${cache}`
      : `⚠ Низкая уверенность: лучше проверить выделенные тайминги вручную${candidate}${cache}`;
  }

  async function inspectSse(response) {
    try {
      const reader = response.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (value) buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          const dataLine = block.split(/\r?\n/).find(line => line.startsWith('data:'));
          if (!dataLine) continue;
          try {
            const data = JSON.parse(dataLine.slice(5).trim());
            if (data.type === 'result') renderQuality(data);
          } catch {}
        }
        if (done) break;
      }
    } catch {}
  }

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input?.url || '';
    const isGeneration = url === '/upload' || url.endsWith('/upload') || url === '/generate' || url.endsWith('/generate');
    if (isGeneration) {
      try {
        if (init.body instanceof FormData) {
          init.body.set('quality_mode', settings.qualityMode);
          init.body.set('use_demucs', settings.demucsMode);
        } else if (typeof init.body === 'string' && (init.headers?.['Content-Type'] === 'application/json' || init.headers?.get?.('Content-Type') === 'application/json')) {
          const body = JSON.parse(init.body || '{}');
          body.quality_mode = settings.qualityMode;
          body.use_demucs = settings.demucsMode;
          init = { ...init, body: JSON.stringify(body) };
        }
      } catch {}
    }
    const response = await originalFetch(input, init);
    if (isGeneration) void inspectSse(response.clone());
    return response;
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addSettings, { once: true });
  } else {
    addSettings();
  }
})();
