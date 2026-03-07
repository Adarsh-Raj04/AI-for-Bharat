import { useState } from "react";

const SOURCE_COLORS = {
  clinical_trial: {
    bg: "bg-violet-100 dark:bg-violet-900/40",
    border: "border-violet-300 dark:border-violet-700",
    dot: "bg-violet-500",
    badge:
      "bg-violet-100 dark:bg-violet-900/60 text-violet-700 dark:text-violet-300",
    label: "Clinical Trial",
  },
  pubmed: {
    bg: "bg-blue-100 dark:bg-blue-900/40",
    border: "border-blue-300 dark:border-blue-700",
    dot: "bg-blue-500",
    badge: "bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300",
    label: "PubMed",
  },
  biorxiv: {
    bg: "bg-emerald-100 dark:bg-emerald-900/40",
    border: "border-emerald-300 dark:border-emerald-700",
    dot: "bg-emerald-500",
    badge:
      "bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300",
    label: "BioRxiv",
  },
  default: {
    bg: "bg-gray-100 dark:bg-gray-800/60",
    border: "border-gray-300 dark:border-gray-600",
    dot: "bg-gray-400",
    badge: "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400",
    label: "Source",
  },
};

const STUDY_TYPE_COLORS = {
  "Phase III": {
    badge:
      "bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300",
  },
  "Phase II": {
    badge:
      "bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300",
  },
  "Phase I": {
    badge:
      "bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300",
  },
  Preprint: {
    badge:
      "bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300",
  },
};

// Flag config — maps flag string → visual treatment
const FLAG_CONFIG = {
  withdrawn: {
    label: "⚠ Withdrawn",
    badge:
      "bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700",
    cardBorder: "border-red-400 dark:border-red-600",
    cardBg: "bg-red-50 dark:bg-red-900/20",
    dot: "bg-red-500",
  },
  retracted: {
    label: "⚠ Retracted",
    badge:
      "bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 border border-red-300 dark:border-red-700",
    cardBorder: "border-red-400 dark:border-red-600",
    cardBg: "bg-red-50 dark:bg-red-900/20",
    dot: "bg-red-500",
  },
  correction: {
    label: "⚠ Correction",
    badge:
      "bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-700",
    cardBorder: "border-amber-400 dark:border-amber-600",
    cardBg: null,
    dot: null,
  },
  preprint: {
    label: "Preprint",
    badge:
      "bg-orange-100 dark:bg-orange-900/50 text-orange-700 dark:text-orange-300",
    cardBorder: null,
    cardBg: null,
    dot: null,
  },
};

const GENERIC_TYPES = new Set([
  "Clinical Trial",
  "Published Study",
  "Research",
]);

// Returns the highest-priority flag override for card styling
function getFlagOverrides(flags) {
  if (!flags || flags.length === 0) return null;
  if (flags.includes("withdrawn")) return FLAG_CONFIG.withdrawn;
  if (flags.includes("retracted")) return FLAG_CONFIG.retracted;
  if (flags.includes("correction")) return FLAG_CONFIG.correction;
  return null;
}

function FlagBadges({ flags }) {
  if (!flags || flags.length === 0) return null;
  return (
    <>
      {flags.map((flag) => {
        const config = FLAG_CONFIG[flag];
        if (!config) return null;
        return (
          <span
            key={flag}
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${config.badge}`}
          >
            {config.label}
          </span>
        );
      })}
    </>
  );
}

function Badges({ item }) {
  const sourceStyle = SOURCE_COLORS[item.source_type] || SOURCE_COLORS.default;
  const phaseStyle = item.type ? STUDY_TYPE_COLORS[item.type] : null;
  const showPhaseBadge = phaseStyle && !GENERIC_TYPES.has(item.type);

  return (
    <div className="flex flex-wrap gap-1 mb-2">
      {/* Flag badges first — highest visual priority */}
      <FlagBadges flags={item.flags} />
      {/* Source type */}
      <span
        className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${sourceStyle.badge}`}
      >
        {sourceStyle.label}
      </span>
      {/* Phase */}
      {showPhaseBadge && (
        <span
          className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${phaseStyle.badge}`}
        >
          {item.type}
        </span>
      )}
    </div>
  );
}

function formatDate(item) {
  if (item.date) {
    return new Date(item.date).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
    });
  }
  return item.year ?? null;
}

// ── Desktop: horizontal ───────────────────────────────────────
function HorizontalTimeline({ timeline }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  return (
    <div className="relative w-full overflow-x-auto pb-4">
      <div className="absolute top-[32px] left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-gray-300 dark:via-gray-600 to-transparent mx-8" />
      <div className="flex gap-3 min-w-max px-4">
        {timeline.map((item, i) => {
          const sourceStyle =
            SOURCE_COLORS[item.source_type] || SOURCE_COLORS.default;
          const flagOverride = getFlagOverrides(item.flags);
          const isHovered = hoveredIdx === i;
          const dateLabel = formatDate(item);
          const dotClass = flagOverride?.dot || sourceStyle.dot;
          const cardBg = flagOverride?.cardBg || sourceStyle.bg;
          const cardBorder = flagOverride?.cardBorder || sourceStyle.border;

          return (
            <div
              key={i}
              className="relative flex flex-col items-center"
              style={{ minWidth: "185px", maxWidth: "220px" }}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {/* Always render to keep dot vertically aligned across all cards */}
              <span className="text-xs font-bold text-gray-500 dark:text-gray-400 mb-2 tracking-widest h-4 block">
                {dateLabel ?? ""}
              </span>
              <div
                className={`w-4 h-4 rounded-full border-2 border-white dark:border-gray-900 ${dotClass} z-10 shadow-md transition-transform duration-200 ${isHovered ? "scale-125" : ""}`}
              />
              <div
                className={`mt-3 rounded-xl border p-3 w-full transition-all duration-200 shadow-sm ${cardBg} ${cardBorder} ${isHovered ? "shadow-md -translate-y-0.5" : ""}`}
              >
                <Badges item={item} />
                <p className="text-xs text-gray-800 dark:text-gray-200 leading-snug line-clamp-3 font-medium">
                  {item.title}
                </p>
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline"
                    onClick={(e) => e.stopPropagation()}
                  >
                    View Study
                    <svg
                      className="w-3 h-3"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Mobile: vertical ──────────────────────────────────────────
function VerticalTimeline({ timeline }) {
  return (
    <div className="relative pl-6">
      <div className="absolute left-2.5 top-0 bottom-0 w-0.5 bg-gradient-to-b from-transparent via-gray-300 dark:via-gray-600 to-transparent" />
      <div className="space-y-4">
        {timeline.map((item, i) => {
          const sourceStyle =
            SOURCE_COLORS[item.source_type] || SOURCE_COLORS.default;
          const flagOverride = getFlagOverrides(item.flags);
          const dateLabel = formatDate(item);
          const dotClass = flagOverride?.dot || sourceStyle.dot;
          const cardBg = flagOverride?.cardBg || sourceStyle.bg;
          const cardBorder = flagOverride?.cardBorder || sourceStyle.border;

          return (
            <div key={i} className="relative flex gap-3">
              <div
                className={`absolute -left-[18px] top-3 w-3.5 h-3.5 rounded-full border-2 border-white dark:border-gray-900 ${dotClass} z-10 shadow`}
              />
              <div
                className={`flex-1 rounded-xl border p-3 shadow-sm ${cardBg} ${cardBorder}`}
              >
                {dateLabel && (
                  <span className="block text-xs font-bold text-gray-500 dark:text-gray-400 tracking-widest mb-1.5">
                    {dateLabel}
                  </span>
                )}
                <Badges item={item} />
                <p className="text-xs text-gray-800 dark:text-gray-200 leading-snug font-medium">
                  {item.title}
                </p>
                {item.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 text-[11px] font-semibold text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    View Study
                    <svg
                      className="w-3 h-3"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                      />
                    </svg>
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────
export default function ResearchTimeline({ timeline }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!timeline || timeline.length === 0) return null;

  const sortedTimeline = [...timeline].sort(
    (a, b) => new Date(a.date || a.year) - new Date(b.date || b.year),
  );

  const activeYears = new Set(sortedTimeline.map((t) => t.year).filter(Boolean))
    .size;
  const flaggedCount = sortedTimeline.filter((t) =>
    t.flags?.some((f) => ["withdrawn", "retracted"].includes(f)),
  ).length;

  return (
    <div className="rounded-xl border border-blue-100 dark:border-blue-800/40 bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm overflow-hidden">
      <button
        onClick={() => setIsExpanded((prev) => !prev)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-blue-50/60 dark:hover:bg-blue-900/20 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-blue-100 dark:bg-blue-900/50">
            <svg
              className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </span>
          <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">
            Research Timeline
          </span>
          <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400">
            {timeline.length} {timeline.length === 1 ? "study" : "studies"}
            {activeYears > 0 &&
              ` • ${activeYears} ${activeYears === 1 ? "year" : "years"}`}
          </span>
          {/* Warn in header if any flagged papers */}
          {flaggedCount > 0 && (
            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400 border border-red-300 dark:border-red-700">
              ⚠ {flaggedCount} flagged
            </span>
          )}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 dark:text-gray-500 transition-transform duration-300 ${isExpanded ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      <div
        className={`transition-all duration-300 ease-in-out overflow-hidden ${isExpanded ? "max-h-[80vh] opacity-100" : "max-h-0 opacity-0"}`}
      >
        <div className="px-4 pb-4 pt-1 border-t border-blue-100 dark:border-blue-800/40 md:overflow-visible overflow-y-auto max-h-[70vh] md:max-h-none">
          <div className="hidden md:block mt-3">
            <HorizontalTimeline timeline={sortedTimeline} />
          </div>
          <div className="md:hidden mt-3">
            <VerticalTimeline timeline={sortedTimeline} />
          </div>
        </div>
      </div>
    </div>
  );
}
