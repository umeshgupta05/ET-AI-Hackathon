import { useState, useRef, useEffect, useCallback } from "react";
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
  Network,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
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
  transcribeVoice,
  updateMe,
  traceWebSocketUrl,
} from "./utils/api";

function AccountDialog({ mode, user, language, onClose, onAuthenticated, onProfileUpdated }) {
  const { t } = useTranslation();
  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [preferredLanguage, setPreferredLanguage] = useState(user?.preferred_language || language);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (mode === "profile") getHistory().then((data) => setHistory(data.items || [])).catch(() => setHistory([]));
  }, [mode]);

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

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="account-dialog" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <div>
            <div className="section-header">{t("secureAccount")}</div>
            <h2>{mode === "login" ? t("login") : mode === "register" ? t("createAccount") : t("profileHistory")}</h2>
          </div>
          <button className="input-btn" onClick={onClose} title={t("close")} aria-label={t("close")}><SvgIcon name="close" /></button>
        </div>
        <form className="account-form" onSubmit={submit}>
          {mode !== "login" && <label>{t("name")}<input value={name} onChange={(e) => setName(e.target.value)} required minLength={2} /></label>}
          {mode !== "profile" && <label>{t("email")}<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>}
          {mode !== "profile" && <label>{t("password")}<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} /></label>}
          {mode !== "login" && <label>{t("preferredLanguage")}<LanguageSelect value={preferredLanguage} onChange={setPreferredLanguage} /></label>}
          {error && <div className="form-error">{error}</div>}
          <button className="primary-command" type="submit" disabled={busy}>{busy ? t("pleaseWait") : mode === "profile" ? t("saveProfile") : mode === "login" ? t("login") : t("createAccount")}</button>
        </form>
        {mode === "profile" && (
          <div className="case-history">
            <div className="section-header">{t("caseHistory")}</div>
            {history.length === 0 ? <p>{t("noCases")}</p> : history.slice(0, 8).map((item) => (
              <div className="history-row" key={item.id}>
                <span>{item.case_type}</span><strong>{item.risk_level}</strong><time>{new Date(item.created_at).toLocaleString()}</time>
              </div>
            ))}
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
  onLanguageChange,
  onShowLogin,
  onShowRegister,
  onShowProfile,
  onLogout,
}) {
  const { t } = useTranslation();
  return (
    <header className="header">
      <div className="header__logo">
        <div className="header__icon">
          <SvgIcon name="shield" />
        </div>
        <div>
          <div className="header__title">{t("appTitle")}</div>
          <div className="header__subtitle">{t("appSubtitle")}</div>
        </div>
      </div>
      <div className="header__actions">
        <div className={`live-status live-status--${liveStatus}`}>
          <span />
          {liveStatus === "live"
            ? t("live")
            : liveStatus === "reconnecting"
              ? t("reconnecting")
              : t("offline")}
        </div>
        <LanguageSelect value={language} onChange={onLanguageChange} />
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
          "Llama 4 Scout",
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
    const sessionId = crypto.randomUUID?.() || String(Date.now());
    const socket = new WebSocket(traceWebSocketUrl(sessionId));
    socket.onopen = () => setLiveStatus("live");
    socket.onerror = () => setLiveStatus("offline");
    socket.onclose = () => setLiveStatus("offline");
    return () => socket.close();
  }, []);

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
          if (transcription.transcript) setInputText(transcription.transcript);
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
    const utterance = new SpeechSynthesisUtterance(
      `${t(`riskLevels.${result.risk_level}`, { defaultValue: result.risk_level || t("unknown") })} ${t("risk")}, ${confidence} ${t("percentConfidence")}. ${reasoning}`,
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
    <div className="app">
      <Header
        systemStatus={systemStatus}
        liveStatus={liveStatus}
        user={user}
        language={language}
        onLanguageChange={changeLanguage}
        onShowLogin={() => setDialogMode("login")}
        onShowRegister={() => setDialogMode("register")}
        onShowProfile={() => setDialogMode("profile")}
        onLogout={() => { logoutUser().catch(() => setAccessToken(null)).finally(() => setUser(null)); }}
      />
      {dialogMode && (
        <AccountDialog
          mode={dialogMode}
          user={user}
          language={language}
          onClose={() => setDialogMode(null)}
          onAuthenticated={setUser}
          onProfileUpdated={(updated) => { setUser(updated); changeLanguage(updated.preferred_language); }}
        />
      )}
      <div className="main">
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

        {/* Right sidebar - Agent Trace */}
        {activeResult && (
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
