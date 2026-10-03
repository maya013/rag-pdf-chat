"use client";

export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "https://rag-pdf-api.onrender.com"
).replace(/\/+$/, "");
export const AUTH_CHANGE_EVENT = "rag-auth-change";
export const AUTH_EXPIRED_EVENT = "rag-auth-expired";
export const AUTH_STORAGE_KEY = "rag-auth-change-v1";
const SESSION_KEY = "rag-browser-session-v1";
const DOCUMENT_KEY = "rag-tab-document-v1";
const CONVERSATION_KEY = "rag-tab-conversation-v1";
export type User = { id: string; email: string };
let accountUser: User | null = null;
let authKnown = false;
let accessToken = "";
let accessExpires = 0;
let tokenRequest: Promise<string> | null = null;
let authVersion = 0;

function getSessionToken(): string {
  let token = localStorage.getItem(SESSION_KEY);
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, token);
    clearSelection();
  }
  return token;
}

export function clearSelection() {
  sessionStorage.removeItem(DOCUMENT_KEY);
  sessionStorage.removeItem(CONVERSATION_KEY);
}

export function setSelection(documentId: string, conversationId: string) {
  sessionStorage.setItem(DOCUMENT_KEY, documentId);
  sessionStorage.setItem(CONVERSATION_KEY, conversationId);
}

export function resetAuthCache() {
  authVersion++;
  authKnown = false;
  accountUser = null;
  accessToken = "";
  accessExpires = 0;
  tokenRequest = null;
  clearSelection();
}

export async function readJson<T>(response: Response): Promise<T> {
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The backend did not return a valid response. Please try again."); }
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail :
      Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).join(" ") :
      "The request failed. Please try again.";
    throw new Error(message);
  }
  return data as T;
}

export async function getCurrentUser(): Promise<User | null> {
  const version = authVersion;
  const data = await readJson<{ user: User | null }>(await window.fetch("/api/auth/me", {
    credentials: "same-origin", cache: "no-store",
  }));
  if (version !== authVersion) throw new Error("Your account changed. Please try again.");
  accountUser = data.user;
  authKnown = true;
  return accountUser;
}

export async function authenticate(
  mode: "signup" | "login", email: string, password: string, importGuest: boolean,
): Promise<{ user: User; imported_documents: number }> {
  const response = await window.fetch(`/api/auth/${mode}`, {
    method: "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json", "X-Session-ID": getSessionToken() },
    body: JSON.stringify({ email, password, import_guest: importGuest }),
  });
  const data = await readJson<{ user: User; imported_documents: number }>(response);
  resetAuthCache();
  accountUser = data.user;
  authKnown = true;
  if (importGuest) localStorage.removeItem(SESSION_KEY);
  localStorage.setItem(AUTH_STORAGE_KEY, crypto.randomUUID());
  window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
  return data;
}

export async function signOut(): Promise<void> {
  await readJson(await window.fetch("/api/auth/logout", {
    method: "POST", credentials: "same-origin", cache: "no-store",
  }));
  // A successful logout starts an empty guest workspace, rather than reopening
  // a previous guest's documents on this browser. Account history stays saved.
  localStorage.removeItem(SESSION_KEY);
  resetAuthCache();
  authKnown = true;
  localStorage.setItem(AUTH_STORAGE_KEY, crypto.randomUUID());
  window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
}

async function getAccessToken(): Promise<string> {
  if (accessToken && Date.now() < accessExpires) return accessToken;
  if (tokenRequest) return tokenRequest;
  const version = authVersion;
  tokenRequest = (async () => {
    const response = await window.fetch("/api/auth/token", {
      credentials: "same-origin", cache: "no-store",
    });
    if (response.status === 401) {
      resetAuthCache();
      window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    }
    const data = await readJson<{ access_token: string; expires_in: number }>(response);
    if (version !== authVersion) throw new Error("Your account changed. Please try again.");
    accessToken = data.access_token;
    accessExpires = Date.now() + Math.max(0, data.expires_in - 30) * 1000;
    return accessToken;
  })();
  const pending = tokenRequest;
  try { return await pending; }
  finally { if (tokenRequest === pending) tokenRequest = null; }
}

// PDF uploads go directly to FastAPI, keeping Vercel's request-size limits out of the upload path.
// Durable login tokens stay in HttpOnly cookies; short-lived API tokens stay only in memory.
export async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  if (typeof window === "undefined") throw new Error("Call apiFetch from the browser.");
  const target = input instanceof Request ? input.url : input.toString();
  const url = new URL(target, window.location.href);
  if (url.origin !== new URL(API_URL).origin) {
    throw new Error("apiFetch can only call your configured PDF API.");
  }
  if (!authKnown) await getCurrentUser();
  const version = authVersion;
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value));
  // Callers cannot inject a durable login token or another browser identity.
  headers.delete("Authorization");
  headers.delete("X-Session-ID");
  if (accountUser) headers.set("Authorization", `Bearer ${await getAccessToken()}`);
  else headers.set("X-Session-ID", getSessionToken());
  const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  const documentId = sessionStorage.getItem(DOCUMENT_KEY);
  const conversationId = sessionStorage.getItem(CONVERSATION_KEY);
  let body = init?.body;
  if (url.pathname === "/" && method === "GET" &&
      !url.searchParams.has("document_id") && !url.searchParams.has("conversation_id")) {
    if (documentId) url.searchParams.set("document_id", documentId);
    if (conversationId) url.searchParams.set("conversation_id", conversationId);
  }
  if (url.pathname === "/ask" && method === "POST") {
    const raw = typeof body === "string" ? body : input instanceof Request ? await input.clone().text() : null;
    if (raw) {
      const payload = JSON.parse(raw);
      if (!payload.document_id && !payload.conversation_id) {
        if (documentId) payload.document_id = documentId;
        if (conversationId) payload.conversation_id = conversationId;
      }
      body = JSON.stringify(payload);
      headers.set("Content-Type", "application/json");
    }
  }
  const requestInput = input instanceof Request ? new Request(url, input) : url;
  const options = { ...init, headers, cache: "no-store" as const,
                    ...(body !== undefined ? { body } : {}) };
  let response = await window.fetch(requestInput instanceof Request ? requestInput.clone() : requestInput, options);
  // A 401 is rejected before backend work begins, so one token refresh is safe.
  if (response.status === 401 && accountUser) {
    accessToken = "";
    accessExpires = 0;
    headers.set("Authorization", `Bearer ${await getAccessToken()}`);
    response = await window.fetch(requestInput instanceof Request ? requestInput.clone() : requestInput, options);
  }
  if (version !== authVersion) throw new Error("Your account changed. Please try again.");
  if (response.ok && (url.pathname === "/" || url.pathname === "/upload" ||
      url.pathname === "/conversations" || url.pathname.endsWith("/messages"))) {
    const data = await response.clone().json();
    if (data.document_id && data.conversation_id) setSelection(data.document_id, data.conversation_id);
  }
  return response;
}
