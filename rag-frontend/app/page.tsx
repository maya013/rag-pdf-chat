"use client";
import { apiFetch as fetch } from "@/lib/api-client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rag-pdf-api.onrender.com";

type Source = {
  page: number;
  score: number;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
};

type RagResponse = {
  answer: string;
  sources: Source[];
};

type UploadResponse = {
  message: string;
  filename: string;
  pages: number;
  chunks: number;
};

const exampleQuestions = [
  "Summarize this document.",
  "What are the most important points?",
  "Are there any important dates or deadlines?",
];

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messageIdRef = useRef(0);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [checkingBackend, setCheckingBackend] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

 useEffect(() => {
  let cancelled = false;

  async function checkBackend() {
    try {
      const response = await fetch(`${API_URL}/`);

      if (!response.ok) {
        throw new Error("Backend check failed.");
      }

      const data = await response.json();

      if (!cancelled) {
        setBackendReady(true);
        setMessages(data.messages ?? []);
      }
    } catch (requestError) {
      console.error(requestError);

      if (!cancelled) {
        setBackendReady(false);
      }
    } finally {
      if (!cancelled) {
        setCheckingBackend(false);
      }
    }
  }

  void checkBackend();

  return () => {
    cancelled = true;
  };
}, []);

  async function uploadPdf(file: File) {
    if (uploading) {
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      return;
    }

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("PDF upload failed.");
      }

      const data: UploadResponse = await response.json();

      setBackendReady(true);
      setMessages([]);
      setNotice(`${data.filename} is ready. Ask it anything.`);
    } catch (requestError) {
      console.error(requestError);
      setError(
        "Could not upload the PDF. The deployed backend may still be waking up.",
      );
    } finally {
      setUploading(false);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function askQuestion(questionText: string) {
    const trimmedQuestion = questionText.trim();

    if (!trimmedQuestion || loading || uploading) {
      return;
    }

    if (!backendReady) {
      setError("The deployed backend is not connected yet.");
      return;
    }

    messageIdRef.current += 1;

    const userMessage: Message = {
      id: `message-${messageIdRef.current}-user`,
      role: "user",
      text: trimmedQuestion,
    };

    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setError("");
    setNotice("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: trimmedQuestion,
        }),
      });

      if (!response.ok) {
        throw new Error("Question request failed.");
      }

      const data: RagResponse = await response.json();

      messageIdRef.current += 1;

      const assistantMessage: Message = {
        id: `message-${messageIdRef.current}-assistant`,
        role: "assistant",
        text: data.answer,
        sources: data.sources,
      };

      setMessages((current) => [...current, assistantMessage]);
    } catch (requestError) {
      console.error(requestError);
      setError(
        "Could not get an answer from the deployed backend. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void askQuestion(question);
  }

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#f7f6f1] text-[#263a33]">
      <div className="pointer-events-none absolute -left-52 -top-52 h-[500px] w-[500px] rounded-full bg-[#dcefe4] blur-3xl" />
      <div className="pointer-events-none absolute -right-52 bottom-0 h-[500px] w-[500px] rounded-full bg-[#ffe5d1] blur-3xl" />

      <section className="relative mx-auto flex min-h-screen w-full max-w-5xl flex-col px-4 py-5 sm:px-6">
        <header className="flex items-center justify-between border-b border-[#dce4df] pb-5">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#315e50] text-xl font-bold text-white shadow-md shadow-[#315e50]/15">
              R
            </div>

            <div>
              <h1 className="text-2xl font-bold tracking-tight text-[#20332d] sm:text-3xl">
                RAG Chat
              </h1>
              <p className="text-xs text-[#75877f]">PDF question assistant</p>
            </div>
          </div>

          <div
            className={`flex items-center gap-2 rounded-full border px-3 py-2 text-xs sm:px-4 sm:text-sm ${
              backendReady
                ? "border-[#bedfc9] bg-[#e5f4ea] text-[#326348]"
                : "border-[#efc5bc] bg-[#fff0ec] text-[#a34e40]"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendReady
                  ? "animate-pulse bg-[#4a9b67]"
                  : "bg-[#d36b58]"
              }`}
            />
            {checkingBackend
              ? "Connecting..."
              : backendReady
                ? "Connected"
                : "Offline"}
          </div>
        </header>

        {notice && (
          <div className="mx-auto mt-5 w-full max-w-3xl rounded-2xl border border-[#bddfc8] bg-[#e5f4ea] px-5 py-3 text-sm text-[#356248]">
            ✓ {notice}
          </div>
        )}

        {error && (
          <div className="mx-auto mt-5 w-full max-w-3xl rounded-2xl border border-[#efc5bc] bg-[#fff0ec] px-5 py-3 text-sm text-[#a34e40]">
            {error}
          </div>
        )}

        <div className="flex flex-1 flex-col py-8">
          {messages.length === 0 ? (
            <div className="m-auto w-full max-w-3xl text-center">
              <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-[#cfe1d6] bg-white/60 px-4 py-2 text-sm text-[#526f63]">
                <span className="text-[#dc775b]">✦</span>
                AI-powered document search
              </div>

              <h2 className="mx-auto max-w-2xl text-3xl font-bold leading-tight tracking-tight text-[#20332d] sm:text-4xl">
                Ask your PDF.
                <span className="text-[#d87357]"> Get a clear answer.</span>
              </h2>

              <p className="mx-auto mt-5 max-w-xl leading-7 text-[#687c74]">
                Press the plus button to upload a document, then ask a question
                about anything inside it.
              </p>

              <div className="mt-10 grid gap-3 sm:grid-cols-3">
                {exampleQuestions.map((example, index) => (
                  <button
                    key={example}
                    type="button"
                    disabled={loading || uploading || !backendReady}
                    onClick={() => void askQuestion(example)}
                    className="group rounded-2xl border border-[#dfe6e1] bg-white/70 p-5 text-left shadow-[0_8px_25px_rgba(61,80,71,0.05)] transition duration-300 hover:-translate-y-1 hover:border-[#accdbb] hover:bg-white hover:shadow-[0_12px_30px_rgba(61,80,71,0.09)] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <span className="mb-4 flex h-8 w-8 items-center justify-center rounded-full bg-[#e3f0e7] text-sm font-bold text-[#3f6b57] transition group-hover:bg-[#315e50] group-hover:text-white">
                      {index + 1}
                    </span>
                    <span className="text-sm leading-6 text-[#53685f]">
                      {example}
                    </span>
                  </button>
                ))}
              </div>

              <p className="mt-6 text-xs text-[#8c9b95]">
                Your PDF can be changed at any time using the + button.
              </p>
            </div>
          ) : (
            <div className="mx-auto flex w-full max-w-3xl flex-col gap-5">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={
                    message.role === "user"
                      ? "ml-auto max-w-[85%] rounded-3xl rounded-br-md bg-[#dcefe4] px-5 py-4 text-[#29483c]"
                      : "mr-auto max-w-[90%] rounded-3xl rounded-bl-md border border-[#dfe6e1] bg-white px-5 py-4 text-[#30463d] shadow-[0_8px_25px_rgba(61,80,71,0.06)]"
                  }
                >
                  <p className="whitespace-pre-wrap leading-7">{message.text}</p>

                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2 border-t border-[#e0e7e2] pt-4">
                      {message.sources.map((source, index) => (
                        <span
                          key={`${source.page}-${index}`}
                          className="rounded-full bg-[#f6ecc0] px-3 py-1 text-xs text-[#746124]"
                        >
                          Page {source.page} · {source.score.toFixed(4)}
                        </span>
                      ))}
                    </div>
                  )}
                </article>
              ))}

              {loading && (
                <div className="mr-auto flex items-center gap-2 rounded-3xl rounded-bl-md border border-[#dfe6e1] bg-white px-5 py-4 text-[#687c74]">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-[#5c9a75]" />
                  <span
                    className="h-2 w-2 animate-bounce rounded-full bg-[#db8265]"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="h-2 w-2 animate-bounce rounded-full bg-[#d3b84f]"
                    style={{ animationDelay: "300ms" }}
                  />
                  <span className="ml-2">Reading your document...</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="sticky bottom-4 mx-auto w-full max-w-3xl">
          <form
            onSubmit={handleSubmit}
            className="flex w-full items-center gap-2 rounded-[26px] border border-[#d5dfd9] bg-white/95 p-3 shadow-[0_18px_50px_rgba(57,78,69,0.14)] backdrop-blur-xl"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];

                if (file) {
                  void uploadPdf(file);
                }
              }}
            />

            <button
              type="button"
              title="Upload a PDF"
              aria-label="Upload a PDF"
              disabled={uploading || !backendReady}
              onClick={() => fileInputRef.current?.click()}
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-[#f5d8c8] text-3xl font-light text-[#a85640] transition duration-300 hover:rotate-90 hover:bg-[#efc5b2] disabled:cursor-not-allowed disabled:opacity-40"
            >
              {uploading ? (
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-[#a85640] border-t-transparent" />
              ) : (
                "+"
              )}
            </button>

            <input
              type="text"
              value={question}
              disabled={loading || uploading || !backendReady}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                uploading
                  ? "Reading your PDF..."
                  : backendReady
                    ? "Ask a question about your PDF..."
                    : "Connecting to the deployed backend..."
              }
              className="min-w-0 flex-1 bg-transparent px-3 py-3 text-[#263a33] outline-none placeholder:text-[#99aaa3] disabled:cursor-not-allowed"
            />

            <button
              type="submit"
              disabled={
                loading || uploading || !question.trim() || !backendReady
              }
              className="rounded-2xl bg-[#315e50] px-6 py-3 font-semibold text-white transition hover:bg-[#254c40] disabled:cursor-not-allowed disabled:bg-[#cbd5d0] disabled:text-[#82928c]"
            >
              {loading ? "Thinking..." : "Ask"}
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}
