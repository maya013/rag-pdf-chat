import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const BACKEND_URL = (
  process.env.RAG_API_URL ?? process.env.NEXT_PUBLIC_API_URL ??
  "https://rag-pdf-api.onrender.com"
).replace(/\/+$/, "");
const COOKIE = process.env.NODE_ENV === "production" ? "__Host-rag-session" : "rag-session";
const actions: Record<string, string> = {
  signup: "POST", login: "POST", logout: "POST", me: "GET", token: "GET",
};
type Context = { params: Promise<{ action: string }> };

function clearCookie(response: NextResponse) {
  response.cookies.set(COOKIE, "", {
    httpOnly: true, secure: process.env.NODE_ENV === "production",
    sameSite: "lax", path: "/", maxAge: 0,
  });
}

async function handle(request: NextRequest, context: Context) {
  const { action } = await context.params;
  if (!actions[action]) return NextResponse.json({ detail: "Not found." }, { status: 404 });
  if (request.method !== actions[action]) {
    return NextResponse.json({ detail: "Method not allowed." }, { status: 405 });
  }
  // Reject cross-site mutations and token reads. Browser POSTs must send Origin.
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");
  let sameOrigin = false;
  if (origin && host) {
    try {
      const parsed = new URL(origin);
      sameOrigin = parsed.host === host &&
        (parsed.protocol === "https:" || (process.env.NODE_ENV !== "production" && parsed.protocol === "http:"));
    } catch { /* An invalid Origin is rejected below. */ }
  }
  if (request.headers.get("sec-fetch-site") === "cross-site" ||
      (origin && !sameOrigin) ||
      (request.method === "POST" && !origin)) {
    return NextResponse.json({ detail: "Request origin is not allowed." }, { status: 403 });
  }
  const session = request.cookies.get(COOKIE)?.value;
  const headers = new Headers({ "Accept": "application/json" });
  if (session) headers.set("Authorization", `Bearer ${session}`);
  const browserToken = request.headers.get("x-session-id");
  if (browserToken && (action === "signup" || action === "login")) {
    headers.set("X-Session-ID", browserToken);
  }
  let body: string | undefined;
  if (action === "signup" || action === "login") {
    if (!request.headers.get("content-type")?.includes("application/json")) {
      return NextResponse.json({ detail: "Send JSON credentials." }, { status: 415 });
    }
    const bytes = await request.text();
    if (bytes.length > 4096) {
      return NextResponse.json({ detail: "Request is too large." }, { status: 413 });
    }
    try {
      const data = JSON.parse(bytes);
      body = JSON.stringify({ email: data.email, password: data.password,
                              import_guest: data.import_guest === true });
    } catch {
      return NextResponse.json({ detail: "Invalid request." }, { status: 400 });
    }
    headers.set("Content-Type", "application/json");
  }
  if (!session && action === "me") {
    return NextResponse.json({ user: null }, { headers: { "Cache-Control": "no-store" } });
  }
  if (!session && action === "token") {
    return NextResponse.json({ detail: "Please log in again." }, { status: 401 });
  }
  if (!session && action === "logout") {
    const response = NextResponse.json({ message: "Logged out." });
    clearCookie(response);
    return response;
  }
  try {
    const upstream = await fetch(`${BACKEND_URL}/auth/${action}`, {
      method: request.method, headers, body, cache: "no-store",
      redirect: "error", signal: AbortSignal.timeout(55000),
    });
    const data = await upstream.json();
    if (action === "me" && upstream.status === 401) {
      const response = NextResponse.json({ user: null }, { headers: { "Cache-Control": "no-store" } });
      clearCookie(response);
      return response;
    }
    // The durable session token stays in an HttpOnly cookie, never in response JSON.
    const sessionToken = data.session_token;
    const expiresIn = data.expires_in;
    delete data.session_token;
    if (action === "login" || action === "signup") delete data.expires_in;
    const response = NextResponse.json(data, {
      status: upstream.status, headers: { "Cache-Control": "no-store" },
    });
    if (upstream.headers.has("retry-after")) {
      response.headers.set("Retry-After", upstream.headers.get("retry-after")!);
    }
    if (upstream.ok && (action === "signup" || action === "login")) {
      if (typeof sessionToken !== "string" || !/^s_[A-Za-z0-9_-]{40,100}$/.test(sessionToken)) {
        return NextResponse.json({ detail: "Invalid login response." }, { status: 502 });
      }
      response.cookies.set(COOKIE, sessionToken, {
        httpOnly: true, secure: process.env.NODE_ENV === "production",
        sameSite: "lax", path: "/", maxAge: Math.min(Number(expiresIn) || 0, 14 * 24 * 60 * 60),
      });
    }
    if ((action === "logout" && upstream.ok) ||
        (upstream.status === 401 && (action === "token" || action === "logout"))) {
      clearCookie(response);
    }
    return response;
  } catch {
    return NextResponse.json({ detail: "Could not reach the backend. Please try again." }, { status: 502 });
  }
}

export const GET = handle;
export const POST = handle;
