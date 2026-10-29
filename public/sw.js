// Cache do app (arquivos estáticos do mesmo domínio). Os modelos ficam no Cache "atlas-models", gerido por src/ai/engine.js.
const C = 'atlas-shell-v1';
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== self.location.origin) return;
  e.respondWith((async () => {
    const c = await caches.open(C);
    const hit = await c.match(e.request);
    const net = fetch(e.request).then((r) => { if (r.status === 200) c.put(e.request, r.clone()); return r; }).catch(() => hit);
    return hit || net;
  })());
});
