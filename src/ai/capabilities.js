export async function detect() {
  const r = { webgpu: false, f16: false, memGB: navigator.deviceMemory ?? null, maxBuffer: null, reason: '' };
  if (!('gpu' in navigator)) { r.reason = 'Este navegador não expõe WebGPU. Use o Chrome atual no Android 12 ou superior.'; return r; }
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) { r.reason = 'Nenhum adaptador WebGPU disponível neste aparelho.'; return r; }
    r.webgpu = true;
    r.f16 = a.features.has('shader-f16');
    r.maxBuffer = a.limits.maxBufferSize;
  } catch (e) { r.reason = `Falha ao iniciar o WebGPU: ${e.message}`; }
  return r;
}
