const DB = 'atlas-image-ai';
const STORE = 'images';
const open = () => new Promise((res, rej) => {
  const r = indexedDB.open(DB, 1);
  r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
  r.onsuccess = () => res(r.result);
  r.onerror = () => rej(r.error);
});
async function tx(mode, fn) {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); res(req.result); };
    t.onerror = () => rej(t.error);
    t.onabort = () => rej(t.error);
  });
}
export const addImage = (rec) => tx('readwrite', (s) => s.add(rec));
export const listImages = () => tx('readonly', (s) => s.getAll());
export const deleteImage = (id) => tx('readwrite', (s) => s.delete(id));
export const clearImages = () => tx('readwrite', (s) => s.clear());
export const storageUsage = async () => (await navigator.storage?.estimate?.()) ?? {};
