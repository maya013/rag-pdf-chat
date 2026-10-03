"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  API_URL, apiFetch, authenticate, getCurrentUser, signOut, readJson,
  clearSelection, setSelection, resetAuthCache,
  AUTH_CHANGE_EVENT, AUTH_EXPIRED_EVENT, AUTH_STORAGE_KEY,
} from "@/lib/api-client";
import type { User } from "@/lib/api-client";

type Source = { page: number; score: number };
type Message = { id: string; role: "user" | "assistant"; text: string; sources?: Source[] };
type Conversation = { id: string; title: string; message_count: number; updated_at: string };
type SavedDocument = { id: string; filename: string; pages: number; conversations: Conversation[] };
type Workspace = {
  document: string | null; document_id?: string; conversation_id?: string;
  pages: number; messages: Message[];
};
type UploadResponse = { filename: string; document_id: string; conversation_id: string; pages: number };
type RagResponse = { answer: string; sources: Source[]; message_id: string };
const examples = ["Summarize this document.", "What are the most important points?", "Are there any important dates or deadlines?"];
const fieldClass = "w-full rounded-xl border border-[#d5dfd9] bg-white px-4 py-3 text-[#263a33] outline-none focus:border-[#315e50] focus:ring-2 focus:ring-[#315e50]/15";
function errorText(error: unknown) { return error instanceof Error ? error.message : "The request failed. Please try again."; }
function shortDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inFlight = useRef(false);
  const workspaceVersion = useRef(0);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<SavedDocument[]>([]);
  const [activeDocument, setActiveDocument] = useState<{ id: string; filename: string; pages: number } | null>(null);
  const [activeConversation, setActiveConversation] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [historySearch, setHistorySearch] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "signup">("signup");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [importGuest, setImportGuest] = useState(true);
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const busy = loading || uploading || switching || authBusy || checkingBackend;

  const refreshHistory = useCallback(async () => {
    const version = workspaceVersion.current;
    try {
      const data = await readJson<{ documents: SavedDocument[] }>(await apiFetch(`${API_URL}/history`));
      if (version === workspaceVersion.current) { setDocuments(data.documents); setHistoryError(""); }
    } catch (requestError) {
      if (version === workspaceVersion.current) setHistoryError(errorText(requestError));
    }
  }, []);

  const reloadWorkspace = useCallback(async () => {
    const version = ++workspaceVersion.current;
    inFlight.current = true;
    setCheckingBackend(true);
    setLoading(false); setUploading(false); setSwitching(false);
    setMessages([]); setDocuments([]); setActiveDocument(null); setActiveConversation("");
    setQuestion(""); setError(""); setHistoryError(""); setBackendReady(false); setUser(null);
    try {
      const currentUser = await getCurrentUser();
      if (version !== workspaceVersion.current) return;
      setUser(currentUser);
      let response = await apiFetch(`${API_URL}/`);
      if (response.status === 404) { clearSelection(); response = await apiFetch(`${API_URL}/`); }
      const data = await readJson<Workspace>(response);
      if (version !== workspaceVersion.current) return;
      setBackendReady(true);
      setMessages(data.messages ?? []);
      if (data.document_id && data.document) {
        setActiveDocument({ id: data.document_id, filename: data.document, pages: data.pages });
        setActiveConversation(data.conversation_id ?? "");
      }
      await refreshHistory();
    } catch (requestError) {
      if (version === workspaceVersion.current) setError(errorText(requestError));
    } finally {
      if (version === workspaceVersion.current) { setCheckingBackend(false); inFlight.current = false; }
    }
  }, [refreshHistory]);

  useEffect(() => {
    void reloadWorkspace();
    const onAuthChange = () => { void reloadWorkspace(); };
    const onStorage = (event: StorageEvent) => {
      if (event.key === AUTH_STORAGE_KEY) { resetAuthCache(); void reloadWorkspace(); }
    };
    const onExpired = () => {
      setNotice("Your login expired. Log in again to reopen your account history.");
      setAuthMode("login"); setAuthOpen(true);
      void reloadWorkspace();
    };
    window.addEventListener(AUTH_CHANGE_EVENT, onAuthChange);
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    window.addEventListener("storage", onStorage);
    return () => {
      workspaceVersion.current++;
      window.removeEventListener(AUTH_CHANGE_EVENT, onAuthChange);
      window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
      window.removeEventListener("storage", onStorage);
    };
  }, [reloadWorkspace]);

  useEffect(() => {
    if (authOpen) { if (!dialogRef.current?.open) dialogRef.current?.showModal(); }
    else dialogRef.current?.close();
  }, [authOpen]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, loading]);

  async function openConversation(document: SavedDocument, conversationId: string) {
    if (inFlight.current || busy) return;
    inFlight.current = true; setSwitching(true); setError(""); setNotice("");
    const version = workspaceVersion.current;
    try {
      const data = await readJson<{ messages: Message[] }>(await apiFetch(
        `${API_URL}/conversations/${conversationId}/messages`,
      ));
      if (version !== workspaceVersion.current) return;
      setSelection(document.id, conversationId);
      setMessages(data.messages); setQuestion("");
      setActiveDocument({ id: document.id, filename: document.filename, pages: document.pages });
      setActiveConversation(conversationId); setSidebarOpen(false);
    } catch (requestError) { if (version === workspaceVersion.current) setError(errorText(requestError)); }
    finally { if (version === workspaceVersion.current) { setSwitching(false); inFlight.current = false; } }
  }

  async function startConversation() {
    if (!activeDocument || inFlight.current || busy) return;
    inFlight.current = true; setSwitching(true); setError(""); setNotice("");
    const version = workspaceVersion.current;
    try {
      const data = await readJson<{ conversation_id: string }>(await apiFetch(`${API_URL}/conversations`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: activeDocument.id }),
      }));
      if (version !== workspaceVersion.current) return;
      setActiveConversation(data.conversation_id); setMessages([]); setQuestion("");
      setNotice("New conversation started with this PDF.");
      await refreshHistory();
    } catch (requestError) { if (version === workspaceVersion.current) setError(errorText(requestError)); }
    finally { if (version === workspaceVersion.current) { setSwitching(false); inFlight.current = false; } }
  }

  async function uploadPdf(file: File) {
    if (inFlight.current || busy) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) { setError("Please select a PDF file."); return; }
    if (file.size > 20 * 1024 * 1024) { setError("Please select a PDF smaller than 20 MB."); return; }
    inFlight.current = true; setUploading(true); setError(""); setNotice("");
    const version = workspaceVersion.current;
    try {
      const formData = new FormData(); formData.append("file", file);
      const data = await readJson<UploadResponse>(await apiFetch(`${API_URL}/upload`, { method: "POST", body: formData }));
      if (version !== workspaceVersion.current) return;
      setActiveDocument({ id: data.document_id, filename: data.filename, pages: data.pages });
      setActiveConversation(data.conversation_id); setMessages([]); setQuestion("");
      setNotice(`${data.filename} is ready. Ask it anything.`);
      await refreshHistory();
    } catch (requestError) { if (version === workspaceVersion.current) setError(errorText(requestError)); }
    finally {
      if (version === workspaceVersion.current) { setUploading(false); inFlight.current = false; }
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function askQuestion(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !activeDocument || !activeConversation || !backendReady || inFlight.current || busy) return;
    inFlight.current = true; setLoading(true); setError(""); setNotice(""); setQuestion("");
    const version = workspaceVersion.current;
    const messageId = crypto.randomUUID();
    setMessages((current) => [...current, { id: messageId, role: "user", text: trimmed }]);
    try {
      const data = await readJson<RagResponse>(await apiFetch(`${API_URL}/ask`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, document_id: activeDocument.id, conversation_id: activeConversation }),
      }));
      if (version !== workspaceVersion.current) return;
      setMessages((current) => [...current, { id: data.message_id, role: "assistant", text: data.answer, sources: data.sources }]);
      await refreshHistory();
    } catch (requestError) {
      if (version === workspaceVersion.current) {
        setMessages((current) => current.filter((message) => message.id !== messageId));
        setQuestion(trimmed); setError(errorText(requestError));
      }
    } finally { if (version === workspaceVersion.current) { setLoading(false); inFlight.current = false; } }
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (authBusy || inFlight.current) return;
    inFlight.current = true; setAuthBusy(true); setAuthError("");
    try {
      const data = await authenticate(authMode, email.trim(), password, importGuest);
      setPassword(""); setAuthOpen(false);
      setNotice(data.imported_documents ? `Welcome! ${data.imported_documents} saved PDF(s) moved into your account.` : "You are logged in. Your history now follows your account.");
    } catch (requestError) { setAuthError(errorText(requestError)); inFlight.current = false; }
    finally { setAuthBusy(false); }
  }

  async function logout() {
    if (inFlight.current || busy) return;
    inFlight.current = true; setAuthBusy(true);
    try { await signOut(); setNotice("Logged out. Account history remains saved."); }
    catch (requestError) { setError(errorText(requestError)); inFlight.current = false; }
    finally { setAuthBusy(false); }
  }

  function showAuth(mode: "signup" | "login") {
    setAuthMode(mode); setAuthError(""); setPassword(""); setAuthOpen(true); setSidebarOpen(false);
  }
  const visibleDocuments = documents.filter((document) =>
    `${document.filename} ${document.conversations.map((conversation) => conversation.title).join(" ")}`
      .toLowerCase().includes(historySearch.toLowerCase()),
  );

  return (
    <main className="relative min-h-screen bg-[#f7f6f1] text-[#263a33]">
      <div className="pointer-events-none fixed -left-52 -top-52 h-[500px] w-[500px] rounded-full bg-[#dcefe4] blur-3xl" />
      <div className="pointer-events-none fixed -right-52 bottom-0 h-[500px] w-[500px] rounded-full bg-[#ffe5d1] blur-3xl" />
      <div className="relative mx-auto max-w-7xl px-4 py-5 sm:px-6">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[#dce4df] pb-5">
          <div className="flex items-center gap-3">
            <button type="button" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="Toggle history" aria-expanded={sidebarOpen}
              className="rounded-xl border border-[#d5dfd9] bg-white/70 px-3 py-2 lg:hidden">☰</button>
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#315e50] text-xl font-bold text-white">R</div>
            <div><h1 className="text-2xl font-bold tracking-tight sm:text-3xl">RAG Chat</h1><p className="text-xs text-[#75877f]">PDF question assistant</p></div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`flex items-center gap-2 rounded-full border px-3 py-2 text-xs ${backendReady ? "border-[#bedfc9] bg-[#e5f4ea] text-[#326348]" : "border-[#efc5bc] bg-[#fff0ec] text-[#a34e40]"}`}>
              <span className={`h-2 w-2 rounded-full ${backendReady ? "bg-[#4a9b67]" : "bg-[#d36b58]"}`} />
              {checkingBackend ? "Connecting..." : backendReady ? "Connected" : "Offline"}
            </span>
            {!user && <button type="button" disabled={busy} onClick={() => showAuth("login")} className="rounded-xl bg-[#315e50] px-4 py-2 text-sm font-medium text-white disabled:opacity-50">Log in</button>}
          </div>
        </header>
        <div className="flex items-start gap-6 pt-6">
          {sidebarOpen && <button type="button" aria-label="Close history" className="fixed inset-0 z-20 bg-black/25 lg:hidden" onClick={() => setSidebarOpen(false)} />}
          <aside aria-label="PDF and conversation history" className={`${sidebarOpen ? "flex" : "hidden"} fixed inset-y-0 left-0 z-30 w-72 flex-col border-r border-[#dce4df] bg-[#f7f6f1] p-5 lg:sticky lg:top-5 lg:z-auto lg:flex lg:max-h-[calc(100vh-3rem)] lg:w-64 lg:shrink-0 lg:rounded-3xl lg:border lg:bg-white/55`}>
            <div className="mb-4 flex items-center justify-between"><h2 className="font-semibold">Your library</h2>
              <button type="button" aria-label="Close history" onClick={() => setSidebarOpen(false)} className="lg:hidden">✕</button>
              <button type="button" disabled={busy || !backendReady} onClick={() => void refreshHistory()} className="hidden text-xs text-[#526f63] disabled:opacity-40 lg:block">Refresh</button>
            </div>
            <button type="button" disabled={busy || !backendReady} onClick={() => fileInputRef.current?.click()} className="mb-3 rounded-xl bg-[#315e50] px-4 py-3 text-sm font-semibold text-white disabled:opacity-40">+ Upload PDF</button>
            <input aria-label="Search history" value={historySearch} onChange={(event) => setHistorySearch(event.target.value)} placeholder="Search your history..." className="mb-4 w-full rounded-xl border border-[#d5dfd9] bg-white/80 px-3 py-2 text-sm outline-none focus:border-[#315e50]" />
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
              {historyError && <p role="alert" className="text-xs text-[#a34e40]">{historyError}</p>}
              {!checkingBackend && !historyError && visibleDocuments.length === 0 && <p className="py-6 text-center text-sm text-[#75877f]">{historySearch ? "No matching history." : "Your PDFs will appear here."}</p>}
              {visibleDocuments.map((document) => (
                <div key={document.id} className={`rounded-2xl border p-3 ${activeDocument?.id === document.id ? "border-[#bddfc8] bg-[#e5f4ea]/70" : "border-[#dfe6e1] bg-white/60"}`}>
                  <p className="break-words text-sm font-semibold">{document.filename}</p>
                  <p className="mt-1 text-xs text-[#75877f]">{document.pages} page{document.pages === 1 ? "" : "s"} · {document.conversations.length} chat{document.conversations.length === 1 ? "" : "s"}</p>
                  <div className="mt-2 space-y-1">{document.conversations.map((conversation) => (
                    <button key={conversation.id} type="button" disabled={busy} onClick={() => void openConversation(document, conversation.id)} aria-current={activeConversation === conversation.id ? "true" : undefined}
                      className={`w-full rounded-xl px-2 py-2 text-left text-xs transition disabled:opacity-40 ${activeConversation === conversation.id ? "bg-[#315e50] text-white" : "text-[#53685f] hover:bg-[#dcefe4]"}`}>
                      <span className="block truncate">{conversation.title}</span><span className="mt-1 block opacity-65">{shortDate(conversation.updated_at)} · {conversation.message_count} messages</span>
                    </button>
                  ))}</div>
                </div>
              ))}
            </div>
            <div className="mt-5 border-t border-[#dce4df] pt-4">
              {user ? <><p className="truncate text-xs font-medium" title={user.email}>{user.email}</p><p className="mt-1 text-xs text-[#75877f]">History saved to your account</p><button type="button" disabled={busy} onClick={() => void logout()} className="mt-3 text-sm text-[#a85640] disabled:opacity-40">Log out</button></> :
                <><p className="text-xs leading-5 text-[#75877f]">Guest history belongs to this browser. Create an account to access it on other devices.</p><button type="button" disabled={busy} onClick={() => showAuth("signup")} className="mt-3 w-full cursor-none rounded-xl bg-[#315e50] px-4 py-3 text-sm font-semibold text-white transition-[transform,background-color,box-shadow] duration-200 ease-out motion-safe:enabled:hover:-translate-y-0.5 motion-safe:enabled:hover:scale-[1.02] enabled:hover:bg-[#254c40] enabled:hover:shadow-[0_8px_20px_rgba(49,94,80,0.2)] motion-safe:enabled:active:translate-y-0 motion-safe:enabled:active:scale-[0.98] enabled:active:bg-[#203f34] enabled:active:shadow-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#315e50] motion-reduce:transition-none disabled:opacity-40">Create account</button></>}
            </div>
          </aside>
          <section aria-label="PDF conversation" className="flex min-h-[calc(100vh-9rem)] min-w-0 flex-1 flex-col">
            {activeDocument && <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[#dfe6e1] bg-white/60 px-4 py-3">
              <div className="min-w-0"><p className="truncate text-sm font-semibold" title={activeDocument.filename}>{activeDocument.filename}</p><p className="text-xs text-[#75877f]">{activeDocument.pages} pages · Saved</p></div>
              <button type="button" disabled={busy} onClick={() => void startConversation()} className="rounded-xl border border-[#cfe1d6] px-3 py-2 text-xs font-medium text-[#315e50] hover:bg-[#e5f4ea] disabled:opacity-40">+ New chat</button>
            </div>}
            {notice && <div role="status" className="mt-4 rounded-2xl border border-[#bddfc8] bg-[#e5f4ea] px-5 py-3 text-sm text-[#356248]">{notice}</div>}
            {error && <div role="alert" className="mt-4 rounded-2xl border border-[#efc5bc] bg-[#fff0ec] px-5 py-3 text-sm text-[#a34e40]">{error}{!backendReady && <button type="button" disabled={busy} onClick={() => void reloadWorkspace()} className="ml-3 underline disabled:opacity-40">Reconnect</button>}</div>}
            <div className="flex flex-1 flex-col py-8">
              {messages.length === 0 ? <div className="m-auto w-full max-w-3xl py-8 text-center">
                <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-[#cfe1d6] bg-white/60 px-4 py-2 text-sm text-[#526f63]"><span className="text-[#dc775b]">✦</span> AI-powered document search</div>
                <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">Ask your PDF.<span className="text-[#d87357]"> Get a clear answer.</span></h2>
                <p className="mx-auto mt-5 max-w-xl leading-7 text-[#687c74]">{activeDocument ? "Your document is ready. Ask a question or choose one below." : "Upload a PDF to start, or reopen a conversation from your library."}</p>
                <div className="mt-8 grid gap-3 sm:grid-cols-3">{examples.map((example, index) => <button key={example} type="button" disabled={busy || !backendReady || !activeDocument} onClick={() => void askQuestion(example)} className="rounded-2xl border border-[#dfe6e1] bg-white/70 p-5 text-left transition hover:-translate-y-1 hover:border-[#accdbb] disabled:cursor-not-allowed disabled:opacity-40"><span className="mb-4 flex h-8 w-8 items-center justify-center rounded-full bg-[#e3f0e7] text-sm font-bold text-[#3f6b57]">{index + 1}</span><span className="text-sm leading-6 text-[#53685f]">{example}</span></button>)}</div>
              </div> : <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
                {messages.map((message) => <article key={message.id} className={message.role === "user" ? "ml-auto max-w-[85%] rounded-3xl rounded-br-md bg-[#dcefe4] px-5 py-4 text-[#29483c]" : "mr-auto max-w-[95%] rounded-3xl rounded-bl-md border border-[#dfe6e1] bg-white px-5 py-4 text-[#30463d] shadow-sm"}>
                  <p className="whitespace-pre-wrap break-words leading-7">{message.text}</p>
                  {!!message.sources?.length && <div className="mt-4 flex flex-wrap gap-2 border-t border-[#e0e7e2] pt-4">{message.sources.map((source, index) => <span key={`${source.page}-${index}`} className="rounded-full bg-[#f6ecc0] px-3 py-1 text-xs text-[#746124]">Page {source.page} · {source.score.toFixed(4)}</span>)}</div>}
                </article>)}
                {loading && <div role="status" className="mr-auto rounded-3xl border border-[#dfe6e1] bg-white px-5 py-4 text-sm text-[#687c74]">Reading your document...</div>}
              </div>}
              <div ref={endRef} />
            </div>
            <div className="sticky bottom-4 mx-auto w-full max-w-3xl">
              <form onSubmit={(event) => { event.preventDefault(); void askQuestion(question); }} className="flex items-center gap-2 rounded-[26px] border border-[#d5dfd9] bg-white/95 p-3 shadow-[0_18px_50px_rgba(57,78,69,0.14)] backdrop-blur-xl">
                <input ref={fileInputRef} type="file" accept=".pdf,application/pdf" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void uploadPdf(file); }} />
                <button type="button" title="Upload a PDF" aria-label="Upload a PDF" disabled={busy || !backendReady} onClick={() => fileInputRef.current?.click()} className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[#f5d8c8] text-3xl text-[#a85640] disabled:opacity-40">{uploading ? "…" : "+"}</button>
                <input type="text" aria-label="Question about your PDF" value={question} disabled={busy || !backendReady || !activeDocument} onChange={(event) => setQuestion(event.target.value)} placeholder={uploading ? "Reading your PDF..." : !activeDocument ? "Upload a PDF to begin..." : "Ask a question about your PDF..."} className="min-w-0 flex-1 bg-transparent px-2 py-3 outline-none placeholder:text-[#99aaa3] disabled:cursor-not-allowed" />
                <button type="submit" disabled={busy || !question.trim() || !backendReady || !activeDocument} className="rounded-2xl bg-[#315e50] px-4 py-3 font-semibold text-white disabled:bg-[#cbd5d0] disabled:text-[#82928c]">{loading ? "Thinking..." : "Ask"}</button>
              </form>
            </div>
          </section>
        </div>
      </div>
      <dialog ref={dialogRef} aria-labelledby="auth-title" onCancel={(event) => { if (authBusy) event.preventDefault(); else setAuthOpen(false); }} onClose={() => setAuthOpen(false)} className="m-auto w-[calc(100%_-_2rem)] max-w-md rounded-3xl border border-[#dce4df] bg-[#f7f6f1] p-7 text-[#263a33] shadow-2xl backdrop:bg-black/35">
        <div className="flex items-center justify-between"><h2 id="auth-title" className="text-2xl font-bold">{authMode === "signup" ? "Your PDFs, anywhere." : "Welcome back."}</h2><button type="button" aria-label="Close login" disabled={authBusy} onClick={() => setAuthOpen(false)} className="px-2 text-xl disabled:opacity-40">×</button></div>
        <p className="mt-3 text-sm leading-6 text-[#687c74]">{authMode === "signup" ? "Create an account to keep your documents and conversations together across devices." : "Log in to reopen your saved documents and conversations."}</p>
        <form onSubmit={(event) => void submitAuth(event)} className="mt-6 space-y-4">
          <label className="block text-sm font-medium">Email<input autoFocus type="email" autoComplete="email" required maxLength={254} value={email} disabled={authBusy} onChange={(event) => setEmail(event.target.value)} className={`mt-2 ${fieldClass}`} /></label>
          <label className="block text-sm font-medium">Password<input type="password" autoComplete={authMode === "signup" ? "new-password" : "current-password"} required minLength={authMode === "signup" ? 12 : 1} maxLength={128} value={password} disabled={authBusy} onChange={(event) => setPassword(event.target.value)} className={`mt-2 ${fieldClass}`} /></label>
          {authMode === "signup" && <p className="text-xs text-[#75877f]">Use at least 12 characters. Longer passwords are welcome.</p>}
          <label className="flex items-start gap-3 text-sm leading-5 text-[#526f63]"><input type="checkbox" checked={importGuest} disabled={authBusy} onChange={(event) => setImportGuest(event.target.checked)} className="mt-1 accent-[#315e50]" />Move this browser&apos;s existing PDF history into my account.</label>
          {authError && <p role="alert" className="rounded-xl bg-[#fff0ec] p-3 text-sm text-[#a34e40]">{authError}</p>}
          <button type="submit" disabled={authBusy || checkingBackend} className="w-full rounded-xl bg-[#315e50] px-4 py-3 font-semibold text-white disabled:opacity-40">{authBusy ? "Please wait..." : authMode === "signup" ? "Create account" : "Log in"}</button>
        </form>
        <p className="mt-5 text-center text-sm text-[#687c74]">{authMode === "signup" ? "Already have an account? " : "New here? "}<button type="button" disabled={authBusy} onClick={() => { setAuthMode(authMode === "signup" ? "login" : "signup"); setAuthError(""); setPassword(""); }} className="font-semibold text-[#315e50] underline">{authMode === "signup" ? "Log in" : "Create account"}</button></p>
      </dialog>
    </main>
  );
}
