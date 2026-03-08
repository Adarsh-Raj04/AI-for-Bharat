import { useState, useRef, useEffect } from "react";
import {
  ChevronDown,
  FileText,
  GitCompare,
  X,
  CheckCircle2,
} from "lucide-react";

export default function ArtifactsMenu({
  docs = [],
  activeFilter,
  compareFilters,
  onSelect,
  onRemoveDoc,
}) {
  const [open, setOpen] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState([]);
  const menuRef = useRef(null);

  // ── Sync internal state when parent clears compareFilters ──────────────────
  // Without this, the menu shows Compare active + both docs ticked even after
  // the user clicks "✕ Clear" outside, because internal state never resets.
  useEffect(() => {
    if (!compareFilters || compareFilters.length === 0) {
      setCompareMode(false);
      setCompareSelection([]);
    } else if (compareFilters.length === 2) {
      setCompareMode(true);
      setCompareSelection(compareFilters);
    }
  }, [compareFilters]);

  // Outside-click closes the dropdown
  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target))
        setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const activeCount = compareFilters ? 2 : activeFilter ? 1 : 0;

  const handleToggleCompare = () => {
    const next = !compareMode;
    setCompareMode(next);
    setCompareSelection([]);
    if (!next) onSelect(activeFilter, null);
  };

  const handleDocClick = (doc) => {
    if (!compareMode) {
      const isSame = activeFilter?.source_id === doc.source_id;
      onSelect(isSame ? null : doc, null);
      setOpen(false);
    } else {
      setCompareSelection((prev) => {
        const already = prev.findIndex((d) => d.source_id === doc.source_id);
        if (already !== -1)
          return prev.filter((d) => d.source_id !== doc.source_id);
        if (prev.length >= 2) return prev;
        const next = [...prev, doc];
        if (next.length === 2) {
          onSelect(null, next);
          setOpen(false);
        }
        return next;
      });
    }
  };

  const handleClearAll = (e) => {
    e.stopPropagation();
    setCompareSelection([]);
    setCompareMode(false);
    onSelect(null, null);
  };

  const isDocSelected = (doc) =>
    compareMode
      ? compareSelection.some((d) => d.source_id === doc.source_id)
      : activeFilter?.source_id === doc.source_id;

  // ── Trigger label ──────────────────────────────────────────────────────────
  let triggerLabel;
  if (compareFilters?.length === 2) {
    triggerLabel = (
      <span className="text-violet-600 dark:text-violet-400 flex items-center gap-1.5">
        <GitCompare className="w-3.5 h-3.5" />
        Comparing 2 docs
      </span>
    );
  } else if (activeFilter) {
    triggerLabel = (
      <span className="text-violet-600 dark:text-violet-400 flex items-center gap-1.5">
        <FileText className="w-3.5 h-3.5" />
        {activeFilter.title.length > 22
          ? activeFilter.title.slice(0, 22) + "…"
          : activeFilter.title}
      </span>
    );
  } else {
    triggerLabel = (
      <span className="flex items-center gap-1.5">
        <FileText className="w-3.5 h-3.5" />
        Uploaded artifacts
      </span>
    );
  }

  return (
    <div ref={menuRef} className="relative flex-shrink-0 w-full sm:w-auto">
      {/* ── Trigger button ── */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`
          w-full sm:w-auto flex items-center justify-between sm:justify-start gap-1.5 whitespace-nowrap select-none transition-colors
          px-3 py-2.5 sm:py-2 rounded-full text-sm font-medium border
          ${
            activeCount > 0
              ? "bg-violet-50 dark:bg-violet-900/30 border-violet-300 dark:border-violet-600 text-violet-700 dark:text-violet-300"
              : "bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:border-gray-400 dark:hover:border-gray-500"
          }
        `}
      >
        {/* Label + badge grouped — never separated */}
        <span className="flex items-center gap-1.5 flex-1">
          {triggerLabel}
          {docs.length > 0 && (
            <span className="flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full text-xs font-bold bg-teal-500 text-white leading-none">
              {docs.length}
            </span>
          )}
        </span>

        {activeCount > 0 ? (
          <span
            onClick={handleClearAll}
            className="hover:text-red-500 transition-colors flex-shrink-0"
          >
            <X className="w-3.5 h-3.5" />
          </span>
        ) : (
          <ChevronDown
            className={`w-3.5 h-3.5 flex-shrink-0 transition-transform ${open ? "rotate-180" : ""}`}
          />
        )}
      </button>

      {/* ── Dropdown ── */}
      {open && (
        <div className="absolute bottom-full mb-2 left-0 right-0 sm:right-auto z-50 w-auto sm:w-72 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-700">
            <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Uploaded Artifacts
            </span>
            <button
              type="button"
              onClick={handleToggleCompare}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                compareMode
                  ? "bg-violet-600 text-white"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              <GitCompare className="w-3 h-3" />
              Compare
            </button>
          </div>

          {/* Compare hint */}
          {compareMode && (
            <div className="px-4 py-2 bg-violet-50 dark:bg-violet-900/20 border-b border-violet-100 dark:border-violet-800/40 text-xs text-violet-600 dark:text-violet-300">
              {compareSelection.length === 0 && "Select 2 documents to compare"}
              {compareSelection.length === 1 &&
                `✓ ${compareSelection[0].title.slice(0, 30)}… — pick one more`}
              {compareSelection.length === 2 && "Both selected"}
            </div>
          )}

          {/* Doc list */}
          {docs.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-gray-400">
              No documents yet.
              <br />
              Use 📎 to upload a PDF or DOCX.
            </div>
          ) : (
            <ul className="max-h-52 overflow-y-auto py-1">
              {docs.map((doc) => {
                const selected = isDocSelected(doc);
                const maxed =
                  compareMode && compareSelection.length >= 2 && !selected;
                return (
                  <li key={doc.source_id}>
                    <button
                      type="button"
                      onClick={() => !maxed && handleDocClick(doc)}
                      disabled={maxed}
                      className={`group w-full flex items-center gap-3 px-4 py-2.5 text-left text-xs transition-colors
                        ${maxed ? "opacity-30 cursor-not-allowed" : "hover:bg-gray-50 dark:hover:bg-gray-800"}
                        ${selected ? "bg-violet-50 dark:bg-violet-900/20" : ""}
                      `}
                    >
                      {selected ? (
                        <CheckCircle2 className="w-4 h-4 text-violet-500 flex-shrink-0" />
                      ) : (
                        <FileText className="w-4 h-4 text-gray-400 flex-shrink-0" />
                      )}
                      <span
                        className={`flex-1 truncate ${selected ? "text-violet-700 dark:text-violet-300 font-medium" : "text-gray-700 dark:text-gray-300"}`}
                      >
                        {doc.title}
                      </span>
                      <span
                        role="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemoveDoc(doc.source_id);
                        }}
                        className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-all p-0.5"
                      >
                        <X className="w-3 h-3" />
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

          <div className="px-4 py-2 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-400">
            Uploaded documents are stored for 24 hours and then automatically
            removed.
          </div>
        </div>
      )}
    </div>
  );
}
