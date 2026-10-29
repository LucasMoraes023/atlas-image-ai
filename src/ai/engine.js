// Pipeline SD-Turbo (1 etapa, 512x512) com ONNX Runtime Web + WebGPU, adaptado do exemplo oficial de guschmue/ort-webgpu.
import * as ort from 'onnxruntime-web/webgpu';
import { AutoTokenizer, env } from '@xenova/transformers';
import { makeRng, randnLatents, toRgba } from './params.js';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = new URL('ort/', document.baseURI).href;
env.allowLocalModels = false;

export const MODEL_BASE = 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main';
const CACHE = 'atlas-models';
const SIGMA = 14.6146;
const VAE_SCALE = 0.18215;
const SIZE = 512;
const SPECS = {
  text_encoder: { mb: 1700, dims: { batch_size: 1 } },
  unet: { mb: 640, dims: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 } },
  vae_decoder: { mb: 95, dims: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } },
};
const urlOf = (n) => `${MODEL_BASE}/${n}/model.onnx`;
const BASE_OPT = {
  executionProviders: ['webgpu'],
  enableMemPattern: false,
  enableCpuMemArena: false,
  preferredOutputLocation: { last_hidden_state: 'gpu-buffer' },
  extra: { session: { disable_prepacking: '1', use_device_allocator_for_initializers: '1', use_ort_model_bytes_directly: '1', use_ort_model_bytes_for_initializers: '1' } },
};

let tokenizer = null;
let sessions = null;

export const tokenizerReady = () => localStorage.getItem('atlas.tok') === '1';

export async function modelsCached() {
  const c = await caches.open(CACHE);
  const hits = await Promise.all(Object.keys(SPECS).map((n) => c.match(urlOf(n))));
  return hits.every(Boolean);
}

async function fetchCached(url, onBytes, signal) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) return hit.arrayBuffer();
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Falha ao baixar o modelo (HTTP ${res.status}).`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes(got, total);
  }
  const blob = new Blob(parts);
  await cache.put(url, new Response(blob));
  return blob.arrayBuffer();
}

export async function load(onStatus, signal) {
  if (sessions) return;
  if (!tokenizer) {
    onStatus('Carregando tokenizador…', null);
    tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
    localStorage.setItem('atlas.tok', '1');
  }
  const s = {};
  for (const [name, spec] of Object.entries(SPECS)) {
    const buf = await fetchCached(urlOf(name), (got, total) => {
      const t = total || spec.mb * 1048576;
      onStatus(`Baixando ${name}: ${(got / 1048576).toFixed(0)} de ${(t / 1048576).toFixed(0)} MB`, got / t);
    }, signal);
    onStatus(`Iniciando ${name} na GPU…`, null);
    s[name] = await ort.InferenceSession.create(buf, { ...BASE_OPT, freeDimensionOverrides: spec.dims });
  }
  sessions = s;
}

export async function generate({ prompt, seed }, signal) {
  const check = () => { if (signal?.aborted) throw new DOMException('Interrompido', 'AbortError'); };
  const { input_ids: ids } = await tokenizer(prompt, { padding: true, max_length: 77, truncation: true, return_tensor: false });
  const { last_hidden_state: hidden } = await sessions.text_encoder.run({ input_ids: new ort.Tensor('int32', Int32Array.from(ids), [1, ids.length]) });
  try {
    check();
    const noise = randnLatents(4 * 64 * 64, SIGMA, makeRng(seed));
    const div = Math.sqrt(SIGMA ** 2 + 1);
    const { out_sample: eps } = await sessions.unet.run({
      sample: new ort.Tensor('float32', noise.map((v) => v / div), [1, 4, 64, 64]),
      timestep: new ort.Tensor('int64', [999n], [1]),
      encoder_hidden_states: hidden,
    });
    check();
    const lat = new Float32Array(noise.length);
    for (let i = 0; i < lat.length; i += 1) lat[i] = (noise[i] - SIGMA * eps.data[i]) / VAE_SCALE;
    const { sample } = await sessions.vae_decoder.run({ latent_sample: new ort.Tensor('float32', lat, [1, 4, 64, 64]) });
    return toRgba(sample.data, SIZE);
  } finally {
    hidden.dispose();
  }
}
