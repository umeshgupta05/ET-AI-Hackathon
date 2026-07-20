import { lazy, Suspense, useState, useRef, useEffect, useCallback } from "react";
import "./CommandCentre.css";
import {
  AlertTriangle,
  BarChart3,
  Bot,
  Camera,
  CheckCircle2,
  Clock3,
  Compass,
  Eye,
  FileAudio,
  Inbox,
  Mic,
  Moon,
  Network,
  Send,
  ScanLine,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Sun,
  UserRound,
  Volume2,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import "./index.css";
import { languages } from "./i18n";
import {
  analyzeMultimodal,
  getDemoTranscript,
  getDemoBenign,
  getAccessToken,
  analyzeTurnByTurn,
  healthCheck,
  getHistory,
  getMe,
  loginUser,
  logoutUser,
  registerUser,
  setAccessToken,
  startRealtimeSession,
  transcribeVoice,
  updateMe,
  traceWebSocketUrl,
} from "./utils/api";

const CommandCentre = lazy(() => import("./CommandCentre"));

const GUEST_HISTORY_KEY = "fraud_shield_guest_history";

function loadGuestHistory() {
  try {
    return JSON.parse(localStorage.getItem(GUEST_HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveGuestHistoryItem(item) {
  const stored = loadGuestHistory();
  localStorage.setItem(GUEST_HISTORY_KEY, JSON.stringify([item, ...stored].slice(0, 50)));
}

function historyMatchesQuery(item, query) {
  if (!query) return true;
  const haystack = [
    item.case_type,
    item.verdict,
    item.risk_level,
    item.query,
    item.summary,
    item.result?.original_text,
    item.result?.text,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function AccountDialog({ mode, user, language, onClose, onAuthenticated, onProfileUpdated, onOpenHistoryItem }) {
  const { t } = useTranslation();
  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [preferredLanguage, setPreferredLanguage] = useState(user?.preferred_language || language);
  const [history, setHistory] = useState([]);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const showHistoryPanel = mode === "profile" || mode === "history";

  useEffect(() => {
    if (!showHistoryPanel) return;
    let cancelled = false;
    setHistoryLoading(true);
    const load = async () => {
      try {
        if (user) {
          const data = await getHistory();
          if (!cancelled) setHistory(data.items || []);
        } else if (!cancelled) {
          setHistory(loadGuestHistory());
        }
      } catch {
        if (!cancelled) setHistory(user ? [] : loadGuestHistory());
      } finally {
        if (!cancelled) setHistoryLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [mode, user, showHistoryPanel]);

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (mode === "profile") {
        const updated = await updateMe({ name, preferred_language: preferredLanguage });
        onProfileUpdated(updated);
      } else {
        const payload = mode === "register"
          ? await registerUser({ name, email, password, preferred_language: preferredLanguage })
          : await loginUser({ email, password });
        setAccessToken(payload.access_token);
        onAuthenticated(payload.user);
      }
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const filteredHistory = history.filter((item) => historyMatchesQuery(item, historyQuery.trim()));
  const dialogTitle =
    mode === "login" ? t("login")
      : mode === "register" ? t("createAccount")
        : mode === "history" ? t("searchHistory")
          : t("profileHistory");

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className={`account-dialog${mode === "history" ? " account-dialog--history" : ""}`}
        role="dialog"
        aria-modal="true"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <div className="section-header">{mode === "history" ? t("caseHistory") : t("secureAccount")}</div>
            <h2>{dialogTitle}</h2>
          </div>
          <button className="input-btn" onClick={onClose} title={t("close")} aria-label={t("close")}><SvgIcon name="close" /></button>
        </div>
        {mode !== "history" && (
          <form className="account-form" onSubmit={submit}>
            {mode !== "login" && <label>{t("name")}<input value={name} onChange={(e) => setName(e.target.value)} required minLength={2} /></label>}
            {mode !== "profile" && <label>{t("email")}<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>}
            {mode !== "profile" && <label>{t("password")}<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} /></label>}
            {mode !== "login" && <label>{t("preferredLanguage")}<LanguageSelect value={preferredLanguage} onChange={setPreferredLanguage} /></label>}
            {error && <div className="form-error">{error}</div>}
            <button className="primary-command" type="submit" disabled={busy}>{busy ? t("pleaseWait") : mode === "profile" ? t("saveProfile") : mode === "login" ? t("login") : t("createAccount")}</button>
          </form>
        )}
        {showHistoryPanel && (
          <div className={`case-history${mode === "history" ? " case-history--standalone" : ""}`}>
            {mode !== "history" && <div className="section-header">{t("caseHistory")}</div>}
            <label className="history-search">
              <span className="sr-only">{t("searchHistory")}</span>
              <input
                type="search"
                value={historyQuery}
                onChange={(e) => setHistoryQuery(e.target.value)}
                placeholder={t("searchHistoryPlaceholder")}
              />
            </label>
            {!user && <p className="history-source-hint">{t("guestHistoryHint")}</p>}
            {user && <p className="history-source-hint">{t("accountHistoryHint")}</p>}
            {historyLoading ? (
              <p>{t("pleaseWait")}</p>
            ) : filteredHistory.length === 0 ? (
              <p>{t("noCases")}</p>
            ) : (
              (mode === "history" ? filteredHistory : filteredHistory.slice(0, 8)).map((item) => (
                <button
                  type="button"
                  className="history-row history-row--action"
                  key={item.id}
                  onClick={() => {
                    if (item.result && onOpenHistoryItem) {
                      onOpenHistoryItem(item);
                      onClose();
                    }
                  }}
                  disabled={!item.result}
                >
                  <span>{item.case_type || t("unknown")}</span>
                  <strong>{item.risk_level || item.verdict || t("unknown")}</strong>
                  {(item.query || item.result?.original_text || item.result?.text) && (
                    <em className="history-row__query">
                      {item.query || item.result?.original_text || item.result?.text}
                    </em>
                  )}
                  <time>{new Date(item.created_at).toLocaleString()}</time>
                </button>
              ))
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function SvgIcon({ name, className = "svg-icon" }) {
  const icons = {
    shield: ShieldCheck,
    inbox: Inbox,
    compass: Compass,
    eye: Eye,
    mic: Mic,
    audio: FileAudio,
    stop: Square,
    listen: Volume2,
    brain: Bot,
    fusion: Network,
    chart: BarChart3,
    settings: Settings2,
    alert: AlertTriangle,
    caution: AlertTriangle,
    check: CheckCircle2,
    camera: Camera,
    user: UserRound,
    send: Send,
    close: X,
    clock: Clock3,
    sparkle: Sparkles,
  };

  const Icon = icons[name] || Settings2;
  return <Icon className={className} aria-hidden="true" strokeWidth={1.9} />;
}

/* ═══════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════ */
function LanguageSelect({ value, onChange, className = "language-select" }) {
  return (
    <select className={className} value={value} onChange={(e) => onChange(e.target.value)}>
      {languages.map((lang) => (
        <option key={lang.code} value={lang.code}>
          {lang.name}
        </option>
      ))}
    </select>
  );
}

function Header({
  systemStatus,
  liveStatus,
  user,
  language,
  theme,
  onThemeToggle,
  onLanguageChange,
  onShowLogin,
  onShowRegister,
  onShowProfile,
  onShowHistory,
  onLogout,
  showLiveScan,
  onOpenLiveScan,
}) {
  const { t } = useTranslation();
  return (
    <header className="header">
      <button className="header__logo header__logo--button" type="button" onClick={onOpenLiveScan} title={t("backToLiveScan")}>
        <div className="header__icon nav-logo-mark">
          <SvgIcon name="shield" />
        </div>
        <div>
          <div className="header__title">{t("appTitle")}</div>
          <div className="header__subtitle">{t("appSubtitle")}</div>
        </div>
      </button>
      <div className="header__actions">
        {showLiveScan && (
          <button className="header__live-scan" type="button" onClick={onOpenLiveScan}>
            <ScanLine size={17} />
            <span>{t("backToLiveScan")}</span>
          </button>
        )}
        <button className="header__live-scan header__history-btn" type="button" onClick={onShowHistory}>
          <SvgIcon name="clock" />
          <span>{t("searchHistory")}</span>
        </button>
        <div className={`live-status live-status--${liveStatus}`}>
          <span />
          {liveStatus === "live"
            ? t("live")
            : liveStatus === "reconnecting"
              ? t("reconnecting")
              : t("offline")}
        </div>
        <LanguageSelect value={language} onChange={onLanguageChange} />
        <button className="theme-toggle" type="button" onClick={onThemeToggle} aria-label={t("toggleTheme", { defaultValue: "Toggle day and night" })}>
          {theme === "dark" ? <Moon size={17} /> : <Sun size={17} />}
        </button>
        <div className="header__badge">
          {systemStatus === "connected" ? t("online") : t("connecting")}
        </div>
        {user ? (
          <div className="account-menu">
            <button className="link-btn" onClick={onShowProfile}>
              <SvgIcon name="user" /> {user.name}
            </button>
            <button className="link-btn" onClick={onLogout}>{t("logout")}</button>
          </div>
        ) : (
          <div className="account-menu">
            <button className="link-btn" onClick={onShowLogin}>{t("login")}</button>
            <button className="link-btn" onClick={onShowRegister}>{t("register")}</button>
          </div>
        )}
      </div>
    </header>
  );
}

function SkylineBackdrop() {
  const buildings = [
    { x: 0, w: 70, h: 130, layer: "far" },
    { x: 60, w: 50, h: 170, layer: "far" },
    { x: 105, w: 85, h: 110, layer: "mid" },
    { x: 185, w: 60, h: 200, layer: "mid" },
    { x: 240, w: 45, h: 140, layer: "far" },
    { x: 280, w: 95, h: 230, layer: "near", flag: true },
    { x: 370, w: 55, h: 160, layer: "mid" },
    { x: 420, w: 75, h: 190, layer: "near" },
    { x: 490, w: 50, h: 120, layer: "far" },
    { x: 535, w: 90, h: 210, layer: "mid" },
    { x: 620, w: 60, h: 260, layer: "near", spire: true },
    { x: 675, w: 70, h: 150, layer: "far" },
    { x: 740, w: 85, h: 195, layer: "mid" },
    { x: 820, w: 55, h: 130, layer: "far" },
    { x: 870, w: 95, h: 220, layer: "near" },
    { x: 960, w: 60, h: 170, layer: "mid" },
    { x: 1015, w: 75, h: 140, layer: "far" },
    { x: 1085, w: 65, h: 200, layer: "near" },
    { x: 1145, w: 55, h: 160, layer: "mid" },
  ];
  const ordered = [...buildings].sort(
    (a, b) =>
      (a.layer === "far" ? 0 : a.layer === "mid" ? 1 : 2) -
      (b.layer === "far" ? 0 : b.layer === "mid" ? 1 : 2),
  );

  return (
    <div className="skyline-backdrop" aria-hidden="true">
      <div className="skyline-sky" />
      <svg className="stars" viewBox="0 0 1200 500" preserveAspectRatio="xMidYMid slice">
        {[120, 260, 380, 520, 640, 760, 880, 960, 1050, 1120, 200, 700].map((cx, i) => (
          <circle key={cx} className="star" cx={cx} cy={[60, 120, 50, 90, 40, 110, 65, 140, 80, 130, 180, 170][i]} r={i % 3 === 0 ? 1.5 : 1} style={{ animationDelay: `${i * 0.22}s` }} />
        ))}
      </svg>
      <div className="sun-moon-wrap">
        <div className="sun-moon-glow" />
        <svg width="100" height="100" viewBox="0 0 100 100">
          <g className="sun-rays" fill="none">
            <line x1="50" y1="8" x2="50" y2="18" />
            <line x1="50" y1="82" x2="50" y2="92" />
            <line x1="8" y1="50" x2="18" y2="50" />
            <line x1="82" y1="50" x2="92" y2="50" />
            <line x1="20" y1="20" x2="27" y2="27" />
            <line x1="73" y1="73" x2="80" y2="80" />
            <line x1="80" y1="20" x2="73" y2="27" />
            <line x1="27" y1="73" x2="20" y2="80" />
          </g>
          <circle className="sun-moon" cx="50" cy="50" r="24" />
        </svg>
      </div>
      <div className="clouds">
        <svg className="cloud cloud-1" width="120" height="40" viewBox="0 0 120 40"><ellipse cx="30" cy="26" rx="26" ry="14" /><ellipse cx="60" cy="18" rx="30" ry="18" /><ellipse cx="90" cy="26" rx="22" ry="12" /></svg>
        <svg className="cloud cloud-2" width="90" height="32" viewBox="0 0 90 32"><ellipse cx="22" cy="20" rx="20" ry="11" /><ellipse cx="46" cy="14" rx="22" ry="14" /><ellipse cx="70" cy="20" rx="18" ry="10" /></svg>
        <svg className="cloud cloud-3" width="100" height="34" viewBox="0 0 100 34"><ellipse cx="24" cy="22" rx="22" ry="12" /><ellipse cx="52" cy="15" rx="26" ry="15" /><ellipse cx="80" cy="22" rx="18" ry="10" /></svg>
      </div>
      <svg className="bird bird-1" width="24" height="12" viewBox="0 0 24 12"><path d="M0 6 Q6 0 12 6 Q18 0 24 6" /></svg>
      <svg className="bird bird-2" width="18" height="9" viewBox="0 0 18 9"><path d="M0 4.5 Q4.5 0 9 4.5 Q13.5 0 18 4.5" /></svg>
      <div className="surveillance-layer">
        {[1, 2].map((n) => (
          <div className={`drone drone-${n}`} key={n}>
            <div className="drone-scan" />
            {n === 1 && <div className="scan-ring" />}
            <svg className="drone-shell" viewBox="0 0 86 50">
              <line className="drone-arm" x1="27" y1="24" x2="10" y2="13" />
              <line className="drone-arm" x1="59" y1="24" x2="76" y2="13" />
              <line className="drone-arm" x1="27" y1="25" x2="10" y2="37" />
              <line className="drone-arm" x1="59" y1="25" x2="76" y2="37" />
              <circle className="rotor-disc" cx="10" cy="12" r="8" />
              <circle className="rotor-disc" cx="76" cy="12" r="8" />
              <circle className="rotor-disc" cx="10" cy="38" r="8" />
              <circle className="rotor-disc" cx="76" cy="38" r="8" />
              <rect className="drone-body" x="25" y="17" width="36" height="18" rx="8" />
              <path className="drone-detail" d="M33 21h20l-4 8H37z" />
              <circle className="drone-lens" cx="43" cy="35" r="3.3" />
            </svg>
          </div>
        ))}
        <div className="cctv cctv-left">
          <div className="cctv-status">AI CCTV LIVE</div>
          <div className="cctv-pole" />
          <div className="cctv-head"><div className="cctv-beam" /><div className="camera-body" /><div className="camera-bracket" /></div>
        </div>
        <div className="cctv cctv-right">
          <div className="cctv-status">SAFE-ZONE 02</div>
          <div className="cctv-pole" />
          <div className="cctv-head"><div className="cctv-beam" /><div className="camera-body" /><div className="camera-bracket" /></div>
        </div>
      </div>
      <div className="patrol-beam" />
      <div className="patrol-dot" />
      <div className="skyline-wrap">
        <svg className="skyline" viewBox="0 0 1200 260" preserveAspectRatio="none">
          {ordered.map((b, bi) => {
            const y = 260 - b.h;
            const cols = Math.max(2, Math.floor(b.w / 14));
            const rows = Math.max(3, Math.floor(b.h / 18));
            const padX = (b.w - cols * 8) / (cols + 1);
            const padY = (b.h - rows * 10) / (rows + 1);
            return (
              <g key={`${b.x}-${b.h}`}>
                <rect x={b.x} y={y} width={b.w} height={b.h} className={`bld-${b.layer}`} />
                {b.spire && <rect x={b.x + b.w / 2 - 2} y={y - 26} width="4" height="26" className={`bld-${b.layer}`} />}
                {b.flag && (
                  <>
                    <rect x={b.x + b.w / 2 - 1} y={y - 22} width="2" height="22" className={`bld-${b.layer}`} />
                    <path d={`M${b.x + b.w / 2 + 1} ${y - 22} L${b.x + b.w / 2 + 15} ${y - 19} L${b.x + b.w / 2 + 1} ${y - 15} Z`} className="bld-near flag" />
                  </>
                )}
                {b.layer !== "far" &&
                  Array.from({ length: rows }).flatMap((_, r) =>
                    Array.from({ length: cols }).map((__, c) => {
                      const lit = (r + c + bi) % 3 !== 0;
                      return (
                        <rect
                          key={`${r}-${c}`}
                          x={b.x + padX + c * (8 + padX)}
                          y={y + padY + r * (10 + padY)}
                          width="6"
                          height="7"
                          rx="0.8"
                          className={lit ? "window lit" : "window"}
                          style={{ animationDelay: `${((r + c + bi) % 5) * 0.45}s` }}
                        />
                      );
                    }),
                  )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   CONFIDENCE GAUGE (SVG)
   ═══════════════════════════════════════════════════════════ */
function ConfidenceGauge({ value = 0, size = 80 }) {
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - value * circumference;
  const color = getSignalColor(value);

  return (
    <div className="confidence-gauge" style={{ width: size, height: size }}>
      <svg className="confidence-gauge__circle" width={size} height={size}>
        <circle
          className="confidence-gauge__bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
        />
        <circle
          className="confidence-gauge__fill"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="confidence-gauge__text" style={{ color }}>
        {Math.round(value * 100)}%
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SCORE BAR
   ═══════════════════════════════════════════════════════════ */
function ScoreBar({ label, value }) {
  return (
    <div className="score-bar">
      <div className="score-bar__header">
        <span className="score-bar__label">{label}</span>
        <span
          className="score-bar__value"
          style={{
            color: getSignalColor(value),
          }}
        >
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="score-bar__track">
        <div
          className={`score-bar__fill score-bar__fill--${value > 0.6 ? "danger" : value > 0.3 ? "accent" : "safe"}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

function getSignalColor(value) {
  return value > 0.7
    ? "var(--signal-alert)"
    : value > 0.4
      ? "var(--signal-caution)"
      : "var(--signal-clear)";
}

function getCaseMeta(result) {
  const seed =
    result?.case_id ||
    result?.id ||
    result?.timestamp ||
    result?.processing_time_seconds ||
    Date.now();
  const numericSeed =
    String(seed).replace(/\D/g, "").slice(-6) ||
    String(Math.abs(JSON.stringify(result || {}).length * 7919)).slice(-6);
  const time = result?.timestamp
    ? new Date(result.timestamp).toLocaleTimeString([], { hour12: false })
    : new Date().toLocaleTimeString([], { hour12: false });
  return `CASE #${numericSeed.padStart(6, "0")} · ${time}`;
}

function getStampText(variant, result, t) {
  const verdict = String(
    result?.verdict || result?.final_verdict || result?.risk_level || "",
  )
    .replace(/_/g, " ")
    .toUpperCase();
  if (verdict) return t(`verdicts.${verdict.toLowerCase().replace(/ /g, "_")}`, { defaultValue: verdict });
  if (variant === "danger") return t("scamConfirmed");
  if (variant === "warning") return t("inconclusive");
  return t("verifiedClear");
}

/* ═══════════════════════════════════════════════════════════
   AGENT TRACE
   ═══════════════════════════════════════════════════════════ */
function AgentTrace({ trace = [] }) {
  const { t } = useTranslation();
  const iconMap = {
    input_detection: { icon: "inbox", cls: "routing" },
    routing: { icon: "compass", cls: "routing" },
    vision_agent: { icon: "eye", cls: "vision" },
    speech_agent: { icon: "mic", cls: "speech" },
    nlp_agent: { icon: "brain", cls: "nlp" },
    fusion: { icon: "fusion", cls: "fusion" },
    calibration: { icon: "chart", cls: "fusion" },
  };

  return (
    <div className="agent-trace">
      <div className="section-header">{t("agentTrace")}</div>
      {trace.map((step, i) => {
        const { icon, cls } = iconMap[step.step] || {
          icon: "settings",
          cls: "routing",
        };
        const score =
          step.confidence ??
          step.fused_score ??
          step.calibrated_score ??
          step.spoof_score;
        const scoreClass =
          score > 0.7 ? "high" : score > 0.4 ? "medium" : "low";

        return (
          <div key={i} className="agent-trace__step">
            <div className={`agent-trace__icon agent-trace__icon--${cls}`}>
              <SvgIcon name={icon} />
            </div>
            <div className="agent-trace__label">
              <strong>{t(`traceSteps.${step.step}`, { defaultValue: step.step?.replace(/_/g, " ") })}</strong>
              {step.reasoning && (
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--text-muted)",
                    marginTop: 2,
                  }}
                >
                  {step.reasoning.slice(0, 80)}
                </div>
              )}
              {step.verdict && (
                <div
                  style={{
                    fontSize: 11,
                    color: "var(--text-primary)",
                    marginTop: 2,
                  }}
                >
                  {t("verdict")}: {t(`verdicts.${step.verdict}`, { defaultValue: step.verdict })}
                </div>
              )}
            </div>
            {score !== undefined && (
              <span
                className={`agent-trace__score agent-trace__score--${scoreClass}`}
              >
                {(score * 100).toFixed(0)}%
              </span>
            )}
            {step.timestamp !== undefined && (
              <span className="agent-trace__time">
                {step.timestamp.toFixed(1)}s
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   VERDICT CARD
   ═══════════════════════════════════════════════════════════ */
function VerdictCard({ result, onListen }) {
  const { t } = useTranslation();
  if (!result) return null;

  const confidence = result.confidence || 0;
  const variant =
    confidence > 0.6 ? "danger" : confidence > 0.3 ? "warning" : "safe";
  const verdictIcon =
    variant === "danger"
      ? "alert"
      : variant === "warning"
        ? "caution"
        : "check";
  const title =
    variant === "danger"
      ? t("highRisk")
      : variant === "warning"
        ? t("suspicious")
        : t("appearsSafe");

  const agentScores = result.fusion_details?.per_agent_scores || {};
  const techniques = [];
  if (result.agent_results?.vision?.techniques_used)
    techniques.push(...result.agent_results.vision.techniques_used);
  if (result.agent_results?.nlp?.techniques_used)
    techniques.push(...result.agent_results.nlp.techniques_used);
  if (result.agent_results?.speech?.techniques_used)
    techniques.push(...result.agent_results.speech.techniques_used);

  return (
    <div className={`verdict-card glass-card verdict-card--${variant}`}>
      <div className="case-eyebrow">{getCaseMeta(result)}</div>
      <div className={`verdict-stamp verdict-stamp--${variant}`}>
        {getStampText(variant, result, t)}
      </div>
      <div className="verdict-card__header">
        <div className="verdict-card__icon">
          <SvgIcon name={verdictIcon} />
        </div>
        <div>
          <div className="verdict-card__title">{title}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {t(`riskLevels.${result.risk_level}`, { defaultValue: result.risk_level })} {t("risk")} | {result.processing_time_seconds}{t("secondsShort")}
          </div>
        </div>
        <ConfidenceGauge value={confidence} />
      </div>
      <button className="verdict-listen" type="button" onClick={() => onListen(result)}>
        <SvgIcon name="listen" /> {t("listenVerdict")}
      </button>
      <div className="verdict-card__body">
        {/* Per-agent scores */}
        {Object.keys(agentScores).length > 0 && (
          <>
            <div className="section-header">{t("perAgentScores")}</div>
            {Object.entries(agentScores).map(([agent, score]) => (
              <ScoreBar
                key={agent}
                label={`${agent.toUpperCase()} ${t("agent")}`}
                value={score}
              />
            ))}
          </>
        )}

        {/* NLP Reasoning */}
        {result.agent_results?.nlp?.reasoning && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">{t("aiReasoning")}</div>
            <div
              style={{
                fontSize: 13,
                color: "var(--text-primary)",
                lineHeight: 1.6,
              }}
            >
              {result.agent_results.nlp.reasoning}
            </div>
          </div>
        )}

        {result.guided_reporting && (
          <div className="reporting-guidance">
            <div className="section-header">
              {t("guidedReporting", { defaultValue: "Guided NCRB Reporting" })}
            </div>
            <div className="reporting-guidance__priority">
              {t("priority", { defaultValue: "Priority" })}: {result.guided_reporting.priority}
            </div>
            <ul className="reporting-guidance__actions">
              {(result.guided_reporting.immediate_actions || []).slice(0, 3).map((action, i) => (
                <li key={i}>{action}</li>
              ))}
            </ul>
            <div className="reporting-guidance__channels">
              {(result.guided_reporting.official_channels || []).map((channel, i) => (
                <span key={i} className="reporting-channel">
                  {channel.url ? (
                    <a href={channel.url} target="_blank" rel="noreferrer">{channel.name}</a>
                  ) : (
                    <strong>{channel.name}</strong>
                  )}
                  {channel.phone && <b>{channel.phone}</b>}
                </span>
              ))}
            </div>
            {result.guided_reporting.note && (
              <p className="reporting-guidance__note">{result.guided_reporting.note}</p>
            )}
          </div>
        )}

        {/* Retrieved pattern matches */}
        {result.agent_results?.nlp?.retrieved_pattern_matches?.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">{t("matchedPatterns")}</div>
            {result.agent_results.nlp.retrieved_pattern_matches
              .slice(0, 2)
              .map((match, i) => (
                <div
                  key={i}
                  style={{
                    padding: "8px 10px",
                    marginTop: 6,
                    borderRadius: 0,
                    background: "var(--ink-750)",
                    border: "1px solid var(--hairline)",
                    fontSize: 12,
                    color: "var(--text-primary)",
                    lineHeight: 1.5,
                  }}
                >
                  <div
                    style={{
                      color: "var(--signal-alert)",
                      fontWeight: 600,
                      fontSize: 11,
                      marginBottom: 2,
                    }}
                  >
                    {match.category?.replace(/_/g, " ")} |{" "}
                    {(match.similarity * 100).toFixed(0)}% {t("match")}
                  </div>
                  {match.pattern?.slice(0, 150)}...
                </div>
              ))}
          </div>
        )}

        {/* Vision attention map */}
        {result.agent_visualizations?.annotated_overlay && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">{t("attentionMap")}</div>
            <div className="attention-map">
              <img
                src={`data:image/png;base64,${result.agent_visualizations.annotated_overlay}`}
                alt={t("attentionAlt")}
              />
              <div className="attention-map__label">
                {t("attentionHint")}
              </div>
            </div>
          </div>
        )}

        {/* Techniques used */}
        {techniques.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">{t("techniquesUsed")}</div>
            <div className="technique-tags">
              {[...new Set(techniques)].map((t, i) => (
                <span key={i} className="technique-tag">
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   SCAM TIMELINE (Confidence Trajectory)
   ═══════════════════════════════════════════════════════════ */
function ScamTimeline({ trajectory = [] }) {
  const { t } = useTranslation();
  if (!trajectory.length) return null;
  const finalConfidence = trajectory[trajectory.length - 1].fused_confidence || 0;
  const finalLabel = finalConfidence >= 0.6
    ? t("timelineHigh")
    : finalConfidence >= 0.3
      ? t("timelineSuspicious")
      : t("timelineClear");

  return (
    <div style={{ marginTop: 12 }}>
      <div className="section-header">{t("confidenceTrajectory")}</div>
      <div className="trajectory-summary">
        <SvgIcon name={finalConfidence >= 0.6 ? "alert" : finalConfidence >= 0.3 ? "caution" : "check"} />
        <strong>{finalLabel}</strong>
        <span>{Math.round(finalConfidence * 100)}% {t("finalConfidence")}</span>
      </div>
      <div className="scam-timeline">
        <div className="scam-timeline__bar">
          {trajectory.map((turnItem, i) => {
            const height = Math.max(turnItem.fused_confidence * 100, 5);
            const color = getSignalColor(turnItem.fused_confidence);
            return (
              <div
                key={i}
                className="scam-timeline__segment"
                style={{ height: `${height}%`, background: color }}
                data-tooltip={`${t("turn")} ${turnItem.turn}: ${(turnItem.fused_confidence * 100).toFixed(0)}%`}
                title={`${t("turn")} ${turnItem.turn}: ${turnItem.turn_text}`}
              />
            );
          })}
        </div>
        <div className="scam-timeline__labels">
          <span>{t("turn")} 1</span>
          <span>{t("turn")} {trajectory.length}</span>
        </div>
      </div>
      {/* Turn details */}
      {trajectory.map((turnItem, i) => (
        <div
          key={i}
          style={{
            padding: "6px 10px",
            marginTop: 4,
            borderRadius: 0,
            background: "var(--ink-750)",
            fontSize: 12,
            borderLeft: `3px solid ${getSignalColor(turnItem.fused_confidence)}`,
          }}
        >
          <span style={{ color: "var(--text-muted)" }}>{t("turn")} {turnItem.turn}:</span>{" "}
          <span style={{ color: "var(--text-primary)" }}>{turnItem.turn_text}</span>
          <span
            style={{
              float: "right",
              fontFamily: "var(--font-mono)",
              fontWeight: 600,
              color:
                turnItem.confidence_delta > 0
                  ? "var(--signal-alert)"
                  : "var(--signal-clear)",
            }}
          >
            {turnItem.confidence_delta > 0 ? "+" : ""}
            {(turnItem.confidence_delta * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   WELCOME SCREEN
   ═══════════════════════════════════════════════════════════ */
function WelcomeScreen({ onDemoScam, onDemoBenign, onDemoImage }) {
  const { t } = useTranslation();
  return (
    <div className="welcome">
      <div className="welcome__icon">
        <SvgIcon name="shield" />
      </div>
      <h1 className="welcome__title">{t("welcomeTitle")}</h1>
      <p className="welcome__desc">{t("welcomeDesc")}</p>
      <div className="welcome__actions">
        <button className="welcome__action-btn" onClick={onDemoScam}>
          <SvgIcon name="alert" /> {t("demoScam")}
        </button>
        <button className="welcome__action-btn" onClick={onDemoBenign}>
          <SvgIcon name="check" /> {t("demoBenign")}
        </button>
        <button className="welcome__action-btn" onClick={onDemoImage}>
          <SvgIcon name="camera" /> {t("uploadImage")}
        </button>
      </div>
      <div
        className="technique-tags"
        style={{ marginTop: 8, justifyContent: "center", maxWidth: 500 }}
      >
        {[
          "Kimi K2.5",
          "GPT-OSS 120B",
          "Qwen 3.6 Vision",
          "YOLOv8",
          "EfficientNet",
          "Grad-CAM",
          "WavLM/AASIST",
          "Whisper",
          "DistilBERT",
          "Hybrid RAG",
          "NPR Analysis",
          "ELA",
          "FFT",
          "Contrastive Learning",
          "Multi-Role CoT",
          "Calibration",
          "CLIP",
          "Ensemble Fusion",
        ].map((t, i) => (
          <span key={i} className="technique-tag">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════ */
export default function App() {
  const { i18n, t } = useTranslation();
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState("connecting");
  const [activeResult, setActiveResult] = useState(null);
  const [, setTrajectory] = useState([]);
  const [language, setLanguage] = useState(i18n.language || "en");
  const [user, setUser] = useState(null);
  const [dialogMode, setDialogMode] = useState(null);
  const [liveStatus, setLiveStatus] = useState("reconnecting");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [viewMode, setViewMode] = useState("analysis");
  const [theme, setTheme] = useState(() => localStorage.getItem("shield_theme") || "light");
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const audioInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Health check on mount
  useEffect(() => {
    healthCheck()
      .then(() => setSystemStatus("connected"))
      .catch(() => setSystemStatus("offline"));
  }, []);

  useEffect(() => {
    if (getAccessToken()) {
      getMe().then(setUser).catch(() => setAccessToken(null));
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = language === "ur" ? "rtl" : "ltr";
  }, [language]);

  useEffect(() => {
    localStorage.setItem("shield_theme", theme);
  }, [theme]);

  useEffect(() => {
    let socket;
    let cancelled = false;
    startRealtimeSession({ channel: "web", language, metadata: { client: "react_app" } })
      .then((session) => {
        if (cancelled) return;
        socket = new WebSocket(traceWebSocketUrl(session.session_id));
        socket.onopen = () => setLiveStatus("live");
        socket.onerror = () => setLiveStatus("offline");
        socket.onclose = () => setLiveStatus("offline");
      })
      .catch(() => setLiveStatus("offline"));
    return () => {
      cancelled = true;
      socket?.close();
    };
  }, [language]);

  const changeLanguage = (nextLanguage) => {
    setLanguage(nextLanguage);
    i18n.changeLanguage(nextLanguage);
    localStorage.setItem("ui_language", nextLanguage);
  };

  const addMessage = useCallback((role, content, data = null) => {
    setMessages((prev) => [
      ...prev,
      { role, content, data, id: Date.now() + Math.random() },
    ]);
  }, []);

  const handleSend = async () => {
    if (!inputText.trim() && !imageFile && !audioFile) return;
    if (isLoading) return;

    const userMsg =
      inputText.trim() ||
      (imageFile
        ? `${t("imageFile")}: ${imageFile.name}`
        : `${t("audioFile")}: ${audioFile.name}`);
    addMessage("user", userMsg);
    setIsLoading(true);
    setInputText("");

    try {
      addMessage("ai", "loading");

      const result = await analyzeMultimodal({
        text: inputText.trim() || undefined,
        image: imageFile || undefined,
        audio: audioFile || undefined,
        language,
      });

      // Remove loading message
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage("ai", "result", result);
      setActiveResult(result);
      
      // Guest History Saving (localStorage); logged-in users are persisted by the API
      if (!user) {
        saveGuestHistoryItem({
          id: Date.now().toString(),
          case_type: result.verdict === "scam" || result.risk_level === "high" || result.risk_level === "critical"
            ? "Fraud Attempt"
            : "Benign Check",
          verdict: result.verdict || result.final_verdict || null,
          risk_level: result.risk_level || (result.verdict === "scam" ? "HIGH" : "LOW"),
          confidence: result.confidence ?? null,
          query: userMsg,
          created_at: new Date().toISOString(),
          result,
        });
      }
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        `${t("error")}: ${error.message}. ${t("backendRequired")}`,
      );
    } finally {
      setIsLoading(false);
      setImageFile(null);
      setAudioFile(null);
    }
  };

  const handleDemoScam = async () => {
    setIsLoading(true);
    try {
      const demo = await getDemoTranscript();
      addMessage(
        "user",
        t("demoScamAnalyzing"),
      );
      addMessage("ai", "loading");

      const result = await analyzeTurnByTurn(demo.turns, language);

      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      setTrajectory(result.trajectory || []);
      addMessage("ai", "trajectory", result);
      if (!user) {
        saveGuestHistoryItem({
          id: Date.now().toString(),
          case_type: "Demo Scam Call",
          verdict: result.final_verdict || result.verdict || null,
          risk_level: result.risk_level || "HIGH",
          confidence: result.confidence ?? null,
          query: t("demoScamAnalyzing"),
          created_at: new Date().toISOString(),
          result,
        });
      }
    } catch {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        t("backendStartHint"),
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoBenign = async () => {
    setIsLoading(true);
    try {
      const demo = await getDemoBenign();
      addMessage("user", t("demoBenignAnalyzing"));
      addMessage("ai", "loading");

      const result = await analyzeTurnByTurn(demo.turns, language);

      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      setTrajectory(result.trajectory || []);
      addMessage("ai", "trajectory", result);
      if (!user) {
        saveGuestHistoryItem({
          id: Date.now().toString(),
          case_type: "Demo Legitimate Call",
          verdict: result.final_verdict || result.verdict || null,
          risk_level: result.risk_level || "LOW",
          confidence: result.confidence ?? null,
          query: t("demoBenignAnalyzing"),
          created_at: new Date().toISOString(),
          result,
        });
      }
    } catch {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        t("backendStartHint"),
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoImage = () => {
    fileInputRef.current?.click();
  };

  const handleRecordVoice = async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordedChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setIsRecording(false);
        const blob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        setIsTranscribing(true);
        try {
          const transcription = await transcribeVoice(blob, language);
          if (transcription.english_transcript || transcription.transcript) {
            setInputText(transcription.english_transcript || transcription.transcript);
          }
        } catch (error) {
          addMessage("ai", `${t("transcriptionFailed")}: ${error.message}`);
        } finally {
          setIsTranscribing(false);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      addMessage("ai", `${t("microphoneUnavailable")}: ${error.message}`);
    }
  };

  const listenToVerdict = (result) => {
    if (!("speechSynthesis" in window)) {
      addMessage("ai", t("speechUnsupported"));
      return;
    }
    window.speechSynthesis.cancel();
    const confidence = Math.round((result.confidence || 0) * 100);
    const reasoning = result.agent_results?.nlp?.reasoning || t("reviewGuidance");
    const guidedActions = result.guided_reporting?.immediate_actions?.slice(0, 2).join(" ");
    const utterance = new SpeechSynthesisUtterance(
      `${t(`riskLevels.${result.risk_level}`, { defaultValue: result.risk_level || t("unknown") })} ${t("risk")}, ${confidence} ${t("percentConfidence")}. ${reasoning} ${guidedActions || ""}`,
    );
    utterance.lang = language;
    window.speechSynthesis.speak(utterance);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app skyline-stage" data-theme={theme}>
      <SkylineBackdrop />
      <Header
        systemStatus={systemStatus}
        liveStatus={liveStatus}
        user={user}
        language={language}
        theme={theme}
        onThemeToggle={() => setTheme((current) => current === "dark" ? "light" : "dark")}
        onLanguageChange={changeLanguage}
        onShowLogin={() => setDialogMode("login")}
        onShowRegister={() => setDialogMode("register")}
        onShowProfile={() => setDialogMode("profile")}
        onShowHistory={() => setDialogMode("history")}
        onLogout={() => { logoutUser().catch(() => setAccessToken(null)).finally(() => setUser(null)); }}
        showLiveScan={viewMode !== "analysis" || messages.length > 0}
        onOpenLiveScan={() => {
          setViewMode("analysis");
          setMessages([]);
          setActiveResult(null);
          setTrajectory([]);
          setImageFile(null);
          setAudioFile(null);
          setInputText("");
          window.scrollTo({ top: 0, behavior: "smooth" });
        }}
      />
      {dialogMode && (
        <AccountDialog
          mode={dialogMode}
          user={user}
          language={language}
          onClose={() => setDialogMode(null)}
          onAuthenticated={setUser}
          onProfileUpdated={(updated) => { setUser(updated); changeLanguage(updated.preferred_language); }}
          onOpenHistoryItem={(item) => {
            setViewMode("analysis");
            setImageFile(null);
            setAudioFile(null);
            setInputText("");
            const query =
              item.query
              || item.result?.original_text
              || item.result?.text
              || item.case_type
              || t("caseHistory");
            const isTrajectory = Boolean(item.result?.trajectory);
            setTrajectory(item.result?.trajectory || []);
            setMessages([
              { role: "user", content: query, data: null, id: Date.now() },
              {
                role: "ai",
                content: isTrajectory ? "trajectory" : "result",
                data: item.result,
                id: Date.now() + 1,
              },
            ]);
            setActiveResult(isTrajectory ? null : item.result);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
      )}
      <div className="view-toggle">
        <button className={viewMode === "analysis" ? "view-btn active" : "view-btn"} onClick={() => setViewMode("analysis")}>
          <ShieldCheck size={16} /> Fraud Analysis
        </button>
        <button className={viewMode === "command" ? "view-btn active" : "view-btn"} onClick={() => setViewMode("command")}>
          <Compass size={16} /> Command Centre
        </button>
      </div>
      <div className="main">
        {viewMode === "command" ? (
          <div className="chat-area" style={{ maxWidth: "100%", flex: 1 }}>
            <Suspense fallback={<div className="cc-loading">{t("agentsAnalyzing")}</div>}>
              <CommandCentre />
            </Suspense>
          </div>
        ) : (
        <div className="chat-area">
          {messages.length === 0 ? (
            <WelcomeScreen
              onDemoScam={handleDemoScam}
              onDemoBenign={handleDemoBenign}
              onDemoImage={handleDemoImage}
            />
          ) : (
            <div className="messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`message message--${msg.role}`}>
                  <div
                    className={`message__avatar message__avatar--${msg.role}`}
                  >
                    <SvgIcon name={msg.role === "ai" ? "shield" : "user"} />
                  </div>
                  <div className="message__content">
                    {msg.content === "loading" ? (
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                        }}
                      >
                        <div className="spinner" />
                        <span
                          style={{ color: "var(--text-muted)", fontSize: 13 }}
                        >
                          {t("agentsAnalyzing")}
                        </span>
                      </div>
                    ) : msg.content === "result" && msg.data ? (
                      <VerdictCard result={msg.data} onListen={listenToVerdict} />
                    ) : msg.content === "trajectory" && msg.data ? (
                      <div>
                        <div
                          style={{
                            fontSize: 14,
                            fontWeight: 600,
                            marginBottom: 8,
                          }}
                        >
                          {t("turnAnalysis")}
                        </div>
                        <ScamTimeline trajectory={msg.data.trajectory || []} />
                      </div>
                    ) : (
                      <div style={{ whiteSpace: "pre-wrap" }}>
                        {msg.content}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Input area */}
          <div className="input-area">
            {(imageFile || audioFile) && (
              <div className="upload-preview">
                {imageFile && (
                  <div className="upload-chip">
                    <SvgIcon name="camera" /> {imageFile.name}
                    <span
                      className="upload-chip__remove"
                      onClick={() => setImageFile(null)}
                    >
                      <SvgIcon name="close" />
                    </span>
                  </div>
                )}
                {audioFile && (
                  <div className="upload-chip">
                    <SvgIcon name="mic" /> {audioFile.name}
                    <span
                      className="upload-chip__remove"
                      onClick={() => setAudioFile(null)}
                    >
                      <SvgIcon name="close" />
                    </span>
                  </div>
                )}
              </div>
            )}
            <div className="input-container">
              <button
                className="input-btn input-btn--upload"
                onClick={() => fileInputRef.current?.click()}
                title={t("uploadImage")}
                aria-label={t("uploadImage")}
              >
                <SvgIcon name="camera" />
              </button>
              <button
                className="input-btn input-btn--upload"
                onClick={() => audioInputRef.current?.click()}
                title={t("uploadAudio")}
                aria-label={t("uploadAudio")}
              >
                <SvgIcon name="audio" />
              </button>
              <button
                className={`input-btn ${isRecording ? "input-btn--recording" : ""}`}
                onClick={handleRecordVoice}
                disabled={isTranscribing}
                title={isRecording ? t("stopRecording") : t("record")}
                aria-label={isRecording ? t("stopRecording") : t("record")}
              >
                <SvgIcon name={isRecording ? "stop" : "mic"} />
              </button>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={isTranscribing ? t("transcribingVoice") : t("inputPlaceholder")}
                rows={1}
              />
              <button
                className="input-btn input-btn--send"
                onClick={handleSend}
                disabled={
                  isLoading || (!inputText.trim() && !imageFile && !audioFile)
                }
                title={t("sendForAnalysis")}
                aria-label={t("sendForAnalysis")}
              >
                {isLoading ? <SvgIcon name="clock" /> : <SvgIcon name="send" />}
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => setImageFile(e.target.files?.[0] || null)}
            />
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              hidden
              onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
            />
          </div>
        </div>

        )}
        {/* Right sidebar - Agent Trace */}
        {viewMode === "analysis" && activeResult && (
          <div className="sidebar">
            <AgentTrace trace={activeResult.trace || []} />
            {activeResult.agents_invoked?.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div className="section-header">{t("agentsInvoked")}</div>
                <div className="technique-tags">
                  {activeResult.agents_invoked.map((a, i) => (
                    <span key={i} className="technique-tag">
                      {a} {t("agent")}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {activeResult.processing_time_seconds && (
              <div
                style={{
                  marginTop: 12,
                  fontSize: 12,
                  color: "var(--text-muted)",
                }}
              >
                {t("totalProcessing")}: {activeResult.processing_time_seconds}{t("secondsShort")}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
