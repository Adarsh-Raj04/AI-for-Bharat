import { useState, useRef } from "react";
import {
  Paperclip,
  FileText,
  X,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { useAuth0 } from "@auth0/auth0-react";

export default function IngestButton({
  onSummaryReady,
  disabled = false,
  sessionId,
}) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // "idle"|"loading"|"success"|"error"
  const [errorMsg, setErrorMsg] = useState("");
  const fileInputRef = useRef(null);
  const { getAccessTokenSilently } = useAuth0();

  const reset = () => {
    setFile(null);
    setStatus("idle");
    setErrorMsg("");
  };

  const close = () => {
    setOpen(false);
    reset();
  };

  const submitFile = async () => {
    if (!file) return;
    setStatus("loading");
    setErrorMsg("");
    try {
      const token = await getAccessTokenSilently();
      const formData = new FormData();
      formData.append("file", file);
      formData.append("session_id", sessionId);
      const res = await fetch("/api/v1/ingest/document", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to process file");
      setStatus("success");
      onSummaryReady({
        summary: data.summary,
        citation: data.citation,
        source_id: data.source_id,
        chunks_indexed: data.chunks_indexed,
        already_existed: data.already_existed,
      });
      setTimeout(close, 1200);
    } catch (err) {
      setStatus("error");
      setErrorMsg(err.message);
    }
  };

  return (
    <div className="relative">
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => {
          if (!disabled) setOpen((o) => !o);
        }}
        disabled={disabled}
        title="Upload PDF or DOCX to summarize"
        className={`
          p-2 rounded-lg transition-colors
          ${
            disabled
              ? "text-gray-300 cursor-not-allowed"
              : "text-gray-400 hover:text-violet-500 hover:bg-violet-50 dark:hover:bg-violet-900/20"
          }
        `}
      >
        <Paperclip className="w-5 h-5" />
      </button>

      {/* Popover */}
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={close} />

          <div
            className="
            absolute bottom-12 left-0 z-40 w-72
            bg-white dark:bg-gray-900
            border border-gray-200 dark:border-gray-700
            rounded-2xl shadow-2xl p-4
          "
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-violet-500" />
                <span className="text-sm font-semibold text-gray-800 dark:text-gray-100">
                  Upload Document
                </span>
              </div>
              <button
                onClick={close}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Drop zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className="
                border-2 border-dashed border-gray-200 dark:border-gray-700
                rounded-xl p-5 text-center cursor-pointer mb-3
                hover:border-violet-400 dark:hover:border-violet-500
                hover:bg-violet-50/50 dark:hover:bg-violet-900/10
                transition-colors
              "
            >
              <FileText className="w-7 h-7 mx-auto mb-2 text-gray-300 dark:text-gray-600" />
              {file ? (
                <p className="text-xs font-medium text-violet-600 dark:text-violet-400 truncate px-2">
                  {file.name}
                </p>
              ) : (
                <>
                  <p className="text-xs font-medium text-gray-600 dark:text-gray-300">
                    Click to choose file
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                    PDF or DOCX · max 20 MB
                  </p>
                </>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc"
              className="hidden"
              onChange={(e) => {
                setFile(e.target.files?.[0] || null);
                setStatus("idle");
                setErrorMsg("");
              }}
            />

            <button
              type="button"
              onClick={submitFile}
              disabled={!file || status === "loading"}
              className={`
                w-full flex items-center justify-center gap-2
                py-2 px-4 rounded-lg text-sm font-medium transition-colors
                ${
                  !file || status === "loading"
                    ? "bg-gray-100 dark:bg-gray-800 text-gray-400 cursor-not-allowed"
                    : "bg-violet-600 hover:bg-violet-700 text-white"
                }
              `}
            >
              {status === "loading" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Processing…
                </>
              ) : (
                "Upload & Summarize"
              )}
            </button>

            {status === "success" && (
              <div className="flex items-center gap-2 mt-3 text-emerald-600 dark:text-emerald-400 text-xs">
                <CheckCircle className="w-4 h-4 flex-shrink-0" />
                <span>Indexed and summarized successfully</span>
              </div>
            )}
            {status === "error" && (
              <div className="flex items-start gap-2 mt-3 text-red-500 dark:text-red-400 text-xs">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
