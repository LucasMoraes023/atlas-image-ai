import { describe, it, expect } from 'vitest';
import { validate, makeRng, randnLatents, toRgba } from '../src/ai/params.js';

describe('validate', () => {
  it('rejeita prompt vazio ou grande demais', () => {
    expect(validate({ prompt: '  ' }).ok).toBe(false);
    expect(validate({ prompt: 'a'.repeat(1001) }).ok).toBe(false);
  });
  it('aceita seed vazia, válida e rejeita inválida', () => {
    expect(validate({ prompt: 'gato', seed: '' }).seed).toBeNull();
    expect(validate({ prompt: 'gato', seed: '42' }).seed).toBe(42);
    expect(validate({ prompt: 'gato', seed: '-1' }).ok).toBe(false);
    expect(validate({ prompt: 'gato', seed: '4294967296' }).ok).toBe(false);
  });
  it('não altera o prompt além do trim', () => {
    expect(validate({ prompt: '  corpo humano, anatomia ' }).prompt).toBe('corpo humano, anatomia');
  });
});
describe('rng e pixels', () => {
  it('mesma seed gera o mesmo ruído', () => {
    expect(randnLatents(16, 1, makeRng(7))).toEqual(randnLatents(16, 1, makeRng(7)));
  });
  it('toRgba limita os valores a 0-255', () => {
    const px = toRgba(new Float32Array([-5, 5, 0]), 1);
    expect([...px]).toEqual([0, 255, 128, 255]);
  });
});
