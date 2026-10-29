// Funções puras: validação, RNG com seed, conversão de tensor em pixels, presets.
export const PROMPT_MAX = 1000;
export const SEED_MAX = 4294967295;

export function validate({ prompt, seed }) {
  const p = String(prompt ?? '').trim();
  if (!p) return { ok: false, error: 'Escreva um prompt.' };
  if (p.length > PROMPT_MAX) return { ok: false, error: `O prompt passa de ${PROMPT_MAX} caracteres.` };
  const s = String(seed ?? '').trim();
  if (s === '') return { ok: true, prompt: p, seed: null };
  if (!/^\d{1,10}$/.test(s) || Number(s) > SEED_MAX) return { ok: false, error: `A seed deve ser um inteiro de 0 a ${SEED_MAX}.` };
  return { ok: true, prompt: p, seed: Number(s) };
}

export function makeRng(seed) { // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randnLatents(n, sigma, rng) { // Box-Muller
  const d = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const u = 1 - rng();
    const v = rng();
    d[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * sigma;
  }
  return d;
}

export function toRgba(chw, size) { // tensor NCHW em [-1,1] -> RGBA
  const hw = size * size;
  const out = new Uint8ClampedArray(hw * 4);
  for (let i = 0; i < hw; i += 1) {
    for (let c = 0; c < 3; c += 1) out[i * 4 + c] = Math.round(Math.min(1, Math.max(0, chw[c * hw + i] / 2 + 0.5)) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export const PRESETS = {
  Fotografia: 'photo, 35mm, natural light, sharp focus, high detail',
  Retrato: 'portrait, soft studio light, shallow depth of field, detailed face',
  Paisagem: 'wide landscape, golden hour, dramatic sky, high detail',
  Anime: 'anime style, clean line art, vibrant colors',
  Ilustração: 'digital illustration, bold shapes, painterly, detailed',
  Fantasia: 'fantasy art, epic, magical atmosphere, intricate details',
  'Arte conceitual': 'concept art, cinematic composition, matte painting',
  'Render 3D': '3d render, octane render, global illumination, ultra detailed',
};
