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
  Inbox,
  Mic,
  Network,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import "./index.css";
import { languages } from "./i18n";
import {
  analyzeMultimodal,
  getDemoTranscript,
  getDemoBenign,
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
            <div className="section-header">Secure account</div>
            <h2>{mode === "login" ? "Log in" : mode === "register" ? "Create account" : "Profile & case history"}</h2>
          </div>
          <button className="input-btn" onClick={onClose} title="Close"><SvgIcon name="close" /></button>
        </div>
        <form className="account-form" onSubmit={submit}>
          {mode !== "login" && <label>Name<input value={name} onChange={(e) => setName(e.target.value)} required minLength={2} /></label>}
          {mode !== "profile" && <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>}
          {mode !== "profile" && <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} /></label>}
          {mode !== "login" && <label>Preferred language<LanguageSelect value={preferredLanguage} onChange={setPreferredLanguage} /></label>}
          {error && <div className="form-error">{error}</div>}
          <button className="primary-command" type="submit" disabled={busy}>{busy ? "Please wait..." : mode === "profile" ? "Save profile" : mode === "login" ? "Log in" : "Create account"}</button>
        </form>
        {mode === "profile" && (
          <div className="case-history">
            <div className="section-header">Case history</div>
            {history.length === 0 ? <p>No saved cases yet.</p> : history.slice(0, 8).map((item) => (
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
function ScoreBar({ label, value, variant = "accent" }) {
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

function getStampText(variant, result) {
  const verdict = String(
    result?.verdict || result?.final_verdict || result?.risk_level || "",
  )
    .replace(/_/g, " ")
    .toUpperCase();
  if (verdict) return verdict;
  if (variant === "danger") return "SCAM CONFIRMED";
  if (variant === "warning") return "INCONCLUSIVE";
  return "VERIFIED CLEAR";
}

/* ═══════════════════════════════════════════════════════════
   AGENT TRACE
   ═══════════════════════════════════════════════════════════ */
function AgentTrace({ trace = [] }) {
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
      <div className="section-header">Agent Execution Trace</div>
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
              <strong>{step.step?.replace(/_/g, " ")}</strong>
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
                  Verdict: {step.verdict}
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
function VerdictCard({ result }) {
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
      ? "HIGH RISK DETECTED"
      : variant === "warning"
        ? "SUSPICIOUS ACTIVITY"
        : "APPEARS SAFE";

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
        {getStampText(variant, result)}
      </div>
      <div className="verdict-card__header">
        <div className="verdict-card__icon">
          <SvgIcon name={verdictIcon} />
        </div>
        <div>
          <div className="verdict-card__title">{title}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {result.risk_level} risk | {result.processing_time_seconds}s
          </div>
        </div>
        <ConfidenceGauge value={confidence} />
      </div>
      <div className="verdict-card__body">
        {/* Per-agent scores */}
        {Object.keys(agentScores).length > 0 && (
          <>
            <div className="section-header">Per-Agent Scores</div>
            {Object.entries(agentScores).map(([agent, score]) => (
              <ScoreBar
                key={agent}
                label={agent.toUpperCase() + " Agent"}
                value={score}
              />
            ))}
          </>
        )}

        {/* NLP Reasoning */}
        {result.agent_results?.nlp?.reasoning && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">AI Reasoning</div>
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
            <div className="section-header">Matched Scam Patterns (RAG)</div>
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
                    {(match.similarity * 100).toFixed(0)}% match
                  </div>
                  {match.pattern?.slice(0, 150)}...
                </div>
              ))}
          </div>
        )}

        {/* Vision attention map */}
        {result.agent_visualizations?.annotated_overlay && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">Attention Map (Grad-CAM)</div>
            <div className="attention-map">
              <img
                src={`data:image/png;base64,${result.agent_visualizations.annotated_overlay}`}
                alt="Attention map"
              />
              <div className="attention-map__label">
                Model attention overlay - red regions are flagged as suspicious
              </div>
            </div>
          </div>
        )}

        {/* Techniques used */}
        {techniques.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="section-header">AI Techniques Used</div>
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
  if (!trajectory.length) return null;

  return (
    <div style={{ marginTop: 12 }}>
      <div className="section-header">Confidence Trajectory</div>
      <div className="scam-timeline">
        <div className="scam-timeline__bar">
          {trajectory.map((t, i) => {
            const height = Math.max(t.fused_confidence * 100, 5);
            const color = getSignalColor(t.fused_confidence);
            return (
              <div
                key={i}
                className="scam-timeline__segment"
                style={{ height: `${height}%`, background: color }}
                data-tooltip={`Turn ${t.turn}: ${(t.fused_confidence * 100).toFixed(0)}%`}
                title={`Turn ${t.turn}: ${t.turn_text}`}
              />
            );
          })}
        </div>
        <div className="scam-timeline__labels">
          <span>Turn 1</span>
          <span>Turn {trajectory.length}</span>
        </div>
      </div>
      {/* Turn details */}
      {trajectory.map((t, i) => (
        <div
          key={i}
          style={{
            padding: "6px 10px",
            marginTop: 4,
            borderRadius: 0,
            background: "var(--ink-750)",
            fontSize: 12,
            borderLeft: `3px solid ${getSignalColor(t.fused_confidence)}`,
          }}
        >
          <span style={{ color: "var(--text-muted)" }}>Turn {t.turn}:</span>{" "}
          <span style={{ color: "var(--text-primary)" }}>{t.turn_text}</span>
          <span
            style={{
              float: "right",
              fontFamily: "var(--font-mono)",
              fontWeight: 600,
              color:
                t.confidence_delta > 0
                  ? "var(--signal-alert)"
                  : "var(--signal-clear)",
            }}
          >
            {t.confidence_delta > 0 ? "+" : ""}
            {(t.confidence_delta * 100).toFixed(0)}%
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
  return (
    <div className="welcome">
      <div className="welcome__icon">
        <SvgIcon name="shield" />
      </div>
      <h1 className="welcome__title">Citizen Fraud Shield</h1>
      <p className="welcome__desc">
        Every year, thousands are trapped on video calls by fake CBI officers,
        and counterfeit notes slip past bank counters undetected. Citizen Fraud
        Shield fuses vision, speech, and language models into one real-time
        verdict, with every decision explainable, calibrated, and audit-ready.
      </p>
      <div className="welcome__actions">
        <button className="welcome__action-btn" onClick={onDemoScam}>
          <SvgIcon name="alert" /> Demo: Scam Call Transcript
        </button>
        <button className="welcome__action-btn" onClick={onDemoBenign}>
          <SvgIcon name="check" /> Demo: Legitimate Call
        </button>
        <button className="welcome__action-btn" onClick={onDemoImage}>
          <SvgIcon name="camera" /> Upload Currency Image
        </button>
      </div>
      <div
        className="technique-tags"
        style={{ marginTop: 8, justifyContent: "center", maxWidth: 500 }}
      >
        {[
          "Groq GPT-OSS",
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
          "XGBoost Stacking",
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
  const { i18n } = useTranslation();
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState("connecting");
  const [activeResult, setActiveResult] = useState(null);
  const [trajectory, setTrajectory] = useState([]);
  const [language, setLanguage] = useState(i18n.language || "en");
  const [user, setUser] = useState(null);
  const [dialogMode, setDialogMode] = useState(null);
  const [liveStatus, setLiveStatus] = useState("reconnecting");
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const audioInputRef = useRef(null);

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
    getMe().then(setUser).catch(() => setAccessToken(null));
  }, []);

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
        ? `Image file: ${imageFile.name}`
        : `Audio file: ${audioFile.name}`);
    addMessage("user", userMsg);
    setIsLoading(true);
    setInputText("");

    try {
      addMessage("ai", "loading");

      const result = await analyzeMultimodal({
        text: inputText.trim() || undefined,
        image: imageFile || undefined,
        audio: audioFile || undefined,
      });

      // Remove loading message
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage("ai", "result", result);
      setActiveResult(result);
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        `Error: ${error.message}. Make sure the backend is running (python main.py)`,
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
        "Demo: Analyzing scam call transcript turn-by-turn...",
      );
      addMessage("ai", "loading");

      const result = await analyzeTurnByTurn(demo.turns);

      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      setTrajectory(result.trajectory || []);
      addMessage("ai", "trajectory", result);
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        `Backend not running. Start with: cd backend && python main.py`,
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoBenign = async () => {
    setIsLoading(true);
    try {
      const demo = await getDemoBenign();
      addMessage("user", "Demo: Analyzing legitimate customer service call...");
      addMessage("ai", "loading");

      const result = await analyzeTurnByTurn(demo.turns);

      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      setTrajectory(result.trajectory || []);
      addMessage("ai", "trajectory", result);
    } catch (error) {
      setMessages((prev) => prev.filter((m) => m.content !== "loading"));
      addMessage(
        "ai",
        `Backend not running. Start with: cd backend && python main.py`,
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoImage = () => {
    fileInputRef.current?.click();
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
                          Agents analyzing...
                        </span>
                      </div>
                    ) : msg.content === "result" && msg.data ? (
                      <VerdictCard result={msg.data} />
                    ) : msg.content === "trajectory" && msg.data ? (
                      <div>
                        <div
                          style={{
                            fontSize: 14,
                            fontWeight: 600,
                            marginBottom: 8,
                          }}
                        >
                          Turn-by-Turn Confidence Analysis
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
                title="Upload image"
              >
                <SvgIcon name="camera" />
              </button>
              <button
                className="input-btn input-btn--upload"
                onClick={() => audioInputRef.current?.click()}
                title="Upload audio"
              >
                <SvgIcon name="mic" />
              </button>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe a suspicious situation, paste a transcript, or upload a file..."
                rows={1}
              />
              <button
                className="input-btn input-btn--send"
                onClick={handleSend}
                disabled={
                  isLoading || (!inputText.trim() && !imageFile && !audioFile)
                }
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
                <div className="section-header">Agents Invoked</div>
                <div className="technique-tags">
                  {activeResult.agents_invoked.map((a, i) => (
                    <span key={i} className="technique-tag">
                      {a} agent
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
                Total processing: {activeResult.processing_time_seconds}s
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
