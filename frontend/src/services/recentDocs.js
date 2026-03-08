/**
 * recentDocs.js — session-scoped recent document store
 *
 * Uses sessionStorage (not localStorage) so docs are scoped to the browser tab.
 * Falls back gracefully if storage is unavailable.
 *
 * Each entry: { source_id, title, timestamp }
 * TTL: 24 hours from last access (timestamp refreshes on re-upload)
 * Max: 10 docs
 */

const STORAGE_KEY = "medresearch_recent_docs";
const ONE_DAY = 24 * 60 * 60 * 1000;
const MAX_DOCS = 10;

function safeRead() {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function safeWrite(docs) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(docs));
  } catch {
    // storage full or unavailable — fail silently
  }
}

/** Returns all non-expired docs, pruning stale entries as a side effect */
export function getRecentDocs() {
  const now = Date.now();
  const valid = safeRead().filter((d) => now - d.timestamp < ONE_DAY);
  safeWrite(valid); // prune stale
  return valid;
}

/**
 * Add or refresh a doc.
 * If source_id already exists, its timestamp is updated to now (extends TTL).
 * New docs are prepended.
 */
export function upsertRecentDoc({ source_id, title }) {
  if (!source_id) return;
  const now = Date.now();
  const docs = safeRead().filter((d) => Date.now() - d.timestamp < ONE_DAY);

  const existingIdx = docs.findIndex((d) => d.source_id === source_id);
  if (existingIdx !== -1) {
    // Refresh timestamp — extends 24h window from now
    docs[existingIdx] = { source_id, title, timestamp: now };
    // Move to front
    const [doc] = docs.splice(existingIdx, 1);
    docs.unshift(doc);
  } else {
    docs.unshift({ source_id, title, timestamp: now });
  }

  safeWrite(docs.slice(0, MAX_DOCS));
}

/** Remove a specific doc (e.g. user dismisses it) */
export function removeRecentDoc(source_id) {
  const docs = safeRead().filter((d) => d.source_id !== source_id);
  safeWrite(docs);
}
