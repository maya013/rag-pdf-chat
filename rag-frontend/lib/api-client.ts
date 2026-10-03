"use client";

// Add this file at src/lib/api-client.ts (or lib/api-client.ts without src/).
const SESSION_KEY = "rag-browser-session-v1";
const DOCUMENT_KEY = "rag-tab-document-v1";
const CONVERSATION_KEY = "rag-tab-conversation-v1";
const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "https://rag-pdf-api.onrender.com"
).replace(/\/+$/, "");

function getSessionToken(): string {
  let token = localStorage.getItem(SESSION_KEY);
  if (!token) {
    token = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, token);
    sessionStorage.removeItem(DOCUMENT_KEY);
    sessionStorage.removeItem(CONVERSATION_KEY);
  }
  return token;
}

// The alias import in page.tsx lets its existing fetch calls use this helper.
export async function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  if (typeof window === "undefined") {
    throw new Error("Call apiFetch from the browser.");
  }

  const target = input instanceof Request ? input.url : input.toString();
  const url = new URL(target, window.location.href);
  if (url.origin !== new URL(API_URL).origin) {
    throw new Error("apiFetch can only call your configured PDF API.");
  }

  const token = getSessionToken();
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  new Headers(init?.headers).forEach((value, key) => headers.set(key, value));
  headers.set("X-Session-ID", token);
  const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  const documentId = sessionStorage.getItem(DOCUMENT_KEY);
  const conversationId = sessionStorage.getItem(CONVERSATION_KEY);
  let body = init?.body;

  if (url.pathname === "/" && method === "GET") {
    if (documentId) url.searchParams.set("document_id", documentId);
    if (conversationId) url.searchParams.set("conversation_id", conversationId);
  }
  if (url.pathname === "/ask" && method === "POST") {
    const rawBody = typeof body === "string"
      ? body
      : input instanceof Request ? await input.clone().text() : null;
    if (rawBody) {
      const payload = JSON.parse(rawBody);
      // Pin requests to the PDF open in this tab, even if another tab uploads.
      if (!payload.document_id && !payload.conversation_id) {
        if (documentId) payload.document_id = documentId;
        if (conversationId) payload.conversation_id = conversationId;
      }
      body = JSON.stringify(payload);
      headers.set("Content-Type", "application/json");
    }
  }

  const requestInput = input instanceof Request ? new Request(url, input) : url;
  const response = await window.fetch(requestInput, { ...init, headers, ...(body !== undefined ? { body } : {}) });
  if (response.ok && (url.pathname === "/" || url.pathname === "/upload")) {
    const data = await response.clone().json();
    if (data.document_id && data.conversation_id) {
      sessionStorage.setItem(DOCUMENT_KEY, data.document_id);
      sessionStorage.setItem(CONVERSATION_KEY, data.conversation_id);
    }
  }
  return response;
}
