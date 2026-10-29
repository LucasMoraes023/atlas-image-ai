// Pipeline SD-Turbo (1 etapa, 512x512) com ONNX Runtime Web + WebGPU, adaptado do exemplo oficial de guschmue/ort-webgpu.
// Memória enxuta: download em streaming direto para o Cache e text encoder carregado só durante a codificação.
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
const mb = (n) => (n / 1048576).toFixed(0);
const BASE_OPT = {
  executionProviders: ['webgpu'],
  enableMemPattern: false,
  enableCpuMemArena: false,
  extra: { session: { disable_prepacking: '1', use_device_allocator_for_initializers: '1', use_ort_model_bytes_directly: '1', use_ort_model_bytes_for_initializers: '1' } },
};

let tokenizer = null;
let unet = null;
let vae = null;
let status = () => {};

export const tokenizerReady = () => localStorage.getItem('atlas.tok') === '1';

export async function modelsCached() {
  const c = await caches.open(CACHE);
  const hits = await Promise.all(Object.keys(SPECS).map((n) => c.match(urlOf(n))));
  return hits.every(Boolean);
}

// Baixa direto para o Cache, com contagem de bytes e contrapressão: nada fica inteiro na memória do JS.
async function download(name, signal) {
  const url = urlOf(name);
  const cache = await caches.open(CACHE);
  if (await cache.match(url)) return;
  const res = await fetch(url, { signal });
  if (!res.ok || !res.body) throw new Error(`Falha ao baixar o modelo (HTTP ${res.status}).`);
  const total = Number(res.headers.get('content-length')) || SPECS[name].mb * 1048576;
  let got = 0;
  const counted = res.body.pipeThrough(new TransformStream({
    transform(chunk, ctl) {
      got += chunk.length;
      status(`Baixando ${name}: ${mb(got)} de ${mb(total)} MB`, Math.min(1, got / total));
      ctl.enqueue(chunk);
    },
  }));
  await cache.put(url, new Response(counted, { headers: { 'content-type': 'application/octet-stream' } }));
}

async function createSession(name) {
  const res = await (await caches.open(CACHE)).match(urlOf(name));
  if (!res) throw new Error(`O modelo ${name} não está no cache.`);
  status(`Iniciando ${name} na GPU…`, null);
  return ort.InferenceSession.create(new Uint8Array(await res.arrayBuffer()), { ...BASE_OPT, freeDimensionOverrides: SPECS[name].dims });
}

export async function load(onStatus, signal) {
  status = onStatus;
  if (!tokenizer) {
    status('Carregando tokenizador…', null);
    tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
    localStorage.setItem('atlas.tok', '1');
  }
  for (const name of Object.keys(SPECS)) await download(name, signal);
  if (!vae) vae = await createSession('vae_decoder');
  if (!unet) unet = await createSession('unet');
}

export async function generate({ prompt, seed }, signal) {
  const check = () => { if (signal?.aborted) throw new DOMException('Interrompido', 'AbortError'); };
  const { input_ids: ids } = await tokenizer(prompt, { padding: true, max_length: 77, truncation: true, return_tensor: false });
  status('Codificando o prompt…', null);
  const te = await createSession('text_encoder'); // só existe durante esta etapa, para poupar memória
  let hidden;
  try {
    const o = (await te.run({ input_ids: new ort.Tensor('int32', Int32Array.from(ids), [1, ids.length]) })).last_hidden_state;
    hidden = new ort.Tensor(o.type, o.data.slice(), o.dims);
  } finally {
    await te.release();
  }
  check();
  status('Gerando a imagem…', null);
  const noise = randnLatents(4 * 64 * 64, SIGMA, makeRng(seed));
  const div = Math.sqrt(SIGMA ** 2 + 1);
  const { out_sample: eps } = await unet.run({
    sample: new ort.Tensor('float32', noise.map((v) => v / div), [1, 4, 64, 64]),
    timestep: new ort.Tensor('int64', [999n], [1]),
    encoder_hidden_states: hidden,
  });
  check();
  const lat = new Float32Array(noise.length);
  for (let i = 0; i < lat.length; i += 1) lat[i] = (noise[i] - SIGMA * eps.data[i]) / VAE_SCALE;
  const { sample } = await vae.run({ latent_sample: new ort.Tensor('float32', lat, [1, 4, 64, 64]) });
  return toRgba(sample.data, SIZE);
}
