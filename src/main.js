import './styles.css';
import { validate, PRESETS, PROMPT_MAX } from './ai/params.js';
import { detect } from './ai/capabilities.js';
import * as engine from './ai/engine.js';
import { addImage, listImages, deleteImage, clearImages, storageUsage } from './gallery/database.js';

const $ = (id) => document.getElementById(id);
const say = (t) => { $('msg').textContent = t; };
const mb = (n) => `${(n / 1048576).toFixed(0)} MB`;
let caps = { webgpu: false, reason: 'Verificando…' };
let ctrl = null;
let busy = false;
let urls = [];

document.querySelectorAll('nav button').forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll('nav button').forEach((x) => x.classList.toggle('on', x === b));
    document.querySelectorAll('main > section').forEach((s) => { s.hidden = s.id !== b.dataset.tab; });
    if (b.dataset.tab === 'gallery') renderGallery();
    if (b.dataset.tab === 'settings') renderSettings();
  };
});

Object.entries(PRESETS).forEach(([name, suffix]) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = name;
  b.title = `Acrescenta: ${suffix}`;
  b.onclick = () => {
    const p = $('prompt');
    const base = p.value.trim();
    p.value = (base ? `${base}, ${suffix}` : suffix).slice(0, PROMPT_MAX);
  };
  $('presets').append(b);
});

function explain(e) {
  const m = String(e?.message ?? e);
  if (e?.name === 'AbortError') return 'Geração interrompida.';
  if (e?.name === 'QuotaExceededError') return 'Sem espaço para guardar o modelo. Libere armazenamento do aparelho.';
  if (/memory|alloc|out of|OOM|device lost/i.test(m)) return 'Memória da GPU insuficiente. Este aparelho provavelmente não suporta este modelo.';
  if (/f16|shader/i.test(m)) return `O aparelho não suporta shader-f16, exigido pelo modelo fp16. (${m})`;
  return `Erro: ${m}`;
}

function setBusy(on) {
  busy = on;
  $('go').hidden = on;
  $('stop').hidden = !on;
  $('bar').hidden = !on;
}

$('stop').onclick = () => ctrl?.abort();
$('go').onclick = async () => {
  if (busy) return;
  const v = validate({ prompt: $('prompt').value, seed: $('seed').value });
  if (!v.ok) return say(v.error);
  if (!caps.webgpu) return say(caps.reason);
  ctrl = new AbortController();
  setBusy(true);
  try {
    await engine.load((t, p) => { say(t); if (p == null) $('bar').removeAttribute('value'); else $('bar').value = p; }, ctrl.signal);
    say('Gerando…');
    $('bar').removeAttribute('value');
    const seed = v.seed ?? Math.floor(Math.random() * 4294967296);
    const t0 = performance.now();
    const rgba = await engine.generate({ prompt: v.prompt, seed }, ctrl.signal);
    const c = $('out');
    c.hidden = false;
    c.getContext('2d').putImageData(new ImageData(rgba, 512, 512), 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    await addImage({ blob, prompt: v.prompt, seed, steps: 1, size: 512, model: 'sd-turbo', createdAt: Date.now() });
    $('seed').value = seed;
    say(`Pronto em ${((performance.now() - t0) / 1000).toFixed(1)} s. Seed ${seed}. Prompt processado: "${v.prompt}"`);
  } catch (e) {
    say(explain(e));
  } finally {
    setBusy(false);
  }
};

async function renderGallery() {
  urls.forEach((u) => URL.revokeObjectURL(u));
  urls = [];
  const grid = $('grid');
  grid.replaceChildren();
  const items = (await listImages()).reverse();
  if (!items.length) { grid.textContent = 'Nenhuma imagem ainda.'; return; }
  items.forEach((it) => {
    const u = URL.createObjectURL(it.blob);
    urls.push(u);
    const card = document.createElement('div');
    card.className = 'card';
    const img = new Image();
    img.src = u;
    img.alt = it.prompt;
    img.onclick = () => window.open(u, '_blank');
    const p = document.createElement('p');
    p.textContent = it.prompt;
    const meta = document.createElement('small');
    meta.textContent = `${new Date(it.createdAt).toLocaleString('pt-BR')} · seed ${it.seed} · ${it.size}px · ${it.steps} etapa`;
    const acts = document.createElement('div');
    acts.className = 'acts';
    const btn = (label, fn) => { const b = document.createElement('button'); b.textContent = label; b.onclick = fn; acts.append(b); };
    btn('Baixar', () => { const a = document.createElement('a'); a.href = u; a.download = `atlas-${it.seed}-${it.id}.png`; a.click(); });
    btn('Reusar', () => { $('prompt').value = it.prompt; $('seed').value = it.seed; document.querySelector('[data-tab=create]').click(); });
    btn('Excluir', async () => { await deleteImage(it.id); renderGallery(); });
    card.append(img, p, meta, acts);
    grid.append(card);
  });
}

async function renderSettings() {
  const cached = await engine.modelsCached();
  const sw = Boolean(await navigator.serviceWorker?.getRegistration());
  const est = await storageUsage();
  const rows = [
    ['WebGPU', caps.webgpu ? 'disponível' : `indisponível. ${caps.reason}`],
    ['shader-f16', caps.f16 ? 'sim' : 'não (o modelo fp16 pode falhar ao iniciar)'],
    ['Memória do aparelho (aprox.)', caps.memGB ? `${caps.memGB} GB` : 'desconhecida'],
    ['Modelo', 'SD-Turbo ONNX (schmuell/sd-turbo-ort-web), 512×512, 1 etapa, ~2,4 GB'],
    ['Modelos em cache', cached ? 'sim' : 'não'],
    ['Offline', cached && sw && engine.tokenizerReady() ? 'pronto' : 'ainda não disponível'],
    ['Armazenamento', est.usage ? `${mb(est.usage)} de ${mb(est.quota)}` : 'indisponível'],
  ];
  $('info').replaceChildren(...rows.map(([k, v]) => { const li = document.createElement('li'); li.textContent = `${k}: ${v}`; return li; }));
}

$('clr-models').onclick = async () => { await caches.delete('atlas-models'); renderSettings(); };
$('clr-gallery').onclick = async () => { if (confirm('Apagar toda a galeria? Não dá para desfazer.')) { await clearImages(); renderSettings(); } };

detect().then((c) => {
  caps = c;
  $('chip').textContent = c.webgpu ? `WebGPU disponível${c.f16 ? '' : ' (sem shader-f16)'}` : c.reason;
});
if ('serviceWorker' in navigator && import.meta.env.PROD) navigator.serviceWorker.register(new URL('sw.js', document.baseURI).href).catch(() => {});
