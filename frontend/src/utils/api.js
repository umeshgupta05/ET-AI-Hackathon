const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

let accessToken = localStorage.getItem("access_token");

export function setAccessToken(token) {
  accessToken = token || null;
  if (accessToken) localStorage.setItem("access_token", accessToken);
  else localStorage.removeItem("access_token");
}

export function getAccessToken() {
  return accessToken;
}

function authHeaders(extra = {}) {
  return accessToken ? { ...extra, Authorization: `Bearer ${accessToken}` } : extra;
}

async function parseOrThrow(response, fallback) {
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: fallback }));
    throw new Error(err.detail || fallback);
  }
  return response.json();
}

export async function registerUser(payload) {
  const response = await fetch(`${API_BASE}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Registration failed");
}

export async function loginUser(payload) {
  const response = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Login failed");
}

export async function getMe() {
  const response = await fetch(`${API_BASE}/api/auth/me`, {
    headers: authHeaders(),
  });
  return parseOrThrow(response, "Profile load failed");
}

export async function logoutUser() {
  const response = await fetch(`${API_BASE}/api/auth/logout`, { method: "POST", headers: authHeaders() });
  if (response.ok) setAccessToken(null);
  return parseOrThrow(response, "Logout failed");
}

export async function updateMe(payload) {
  const response = await fetch(`${API_BASE}/api/auth/me`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Profile update failed");
}

export async function getHistory(query = "") {
  const url = query.trim()
    ? `${API_BASE}/api/history?query=${encodeURIComponent(query.trim())}`
    : `${API_BASE}/api/history`;
  const response = await fetch(url, {
    headers: authHeaders(),
  });
  return parseOrThrow(response, "History load failed");
}

export async function analyzeMultimodal({ text, image, audio, language = "en" }) {
  const formData = new FormData();
  if (text) formData.append("text", text);
  if (image) formData.append("image", image);
  if (audio) formData.append("audio", audio);
  formData.append("language", language);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  return parseOrThrow(response, "Analysis failed");
}

export async function analyzeText(text, language = "en") {
  const response = await fetch(`${API_BASE}/api/analyze/text`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ text, language }),
  });
  return parseOrThrow(response, "Text analysis failed");
}

export async function analyzeTurnByTurn(turns, language = "en") {
  const response = await fetch(`${API_BASE}/api/analyze/turns`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ turns, language }),
  });

  return parseOrThrow(response, "Turn-by-turn analysis failed");
}

export async function transcribeVoice(audio, language = "en") {
  const formData = new FormData();
  formData.append("audio", audio, "voice.webm");
  formData.append("language", language);
  const response = await fetch(`${API_BASE}/api/voice/transcribe`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  return parseOrThrow(response, "Voice transcription failed");
}

export async function getLanguages() {
  const response = await fetch(`${API_BASE}/api/languages`);
  return parseOrThrow(response, "Language catalog unavailable");
}

export async function getGraphAnalysis() {
  const response = await fetch(`${API_BASE}/api/graph/analyze`);
  return parseOrThrow(response, "Graph analysis unavailable");
}

export async function getGraphVisualization() {
  const response = await fetch(`${API_BASE}/api/graph/visualization`);
  return parseOrThrow(response, "Graph visualization unavailable");
}

export async function getDemoTranscript() {
  const response = await fetch(`${API_BASE}/api/demo/scam-transcript`);
  return response.json();
}

export async function getDemoBenign() {
  const response = await fetch(`${API_BASE}/api/demo/benign-transcript`);
  return response.json();
}

export async function healthCheck() {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}

export async function getHotspots(latitude, longitude) {
  const query = Number.isFinite(latitude) && Number.isFinite(longitude)
    ? `?latitude=${latitude}&longitude=${longitude}`
    : "";
  const response = await fetch(`${API_BASE}/api/intelligence/hotspots${query}`);
  return parseOrThrow(response, "Hotspot intelligence unavailable");
}

export async function getReportingGuidance(riskLevel = "medium") {
  const response = await fetch(`${API_BASE}/api/reporting/guidance?risk_level=${encodeURIComponent(riskLevel)}`);
  return parseOrThrow(response, "Reporting guidance unavailable");
}

export async function getEvidencePackage(caseId) {
  const response = await fetch(`${API_BASE}/api/cases/${caseId}/evidence`, {
    headers: authHeaders(),
  });
  return parseOrThrow(response, "Evidence package unavailable");
}

export function traceWebSocketUrl(sessionId) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${protocol}//${host}/ws/session/${sessionId}`;
}

export async function startRealtimeSession({ channel = "web", language = "en", metadata = {} } = {}) {
  const response = await fetch(`${API_BASE}/api/realtime/sessions`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ channel, language, metadata }),
  });
  return parseOrThrow(response, "Realtime session unavailable");
}

export async function getThreatFeed() {
  const response = await fetch(`${API_BASE}/api/intelligence/threat-feed`);
  return parseOrThrow(response, "Threat feed unavailable");
}

export async function getCommandCentre(latitude, longitude) {
  const query = Number.isFinite(latitude) && Number.isFinite(longitude)
    ? `?latitude=${latitude}&longitude=${longitude}`
    : "";
  const response = await fetch(`${API_BASE}/api/intelligence/command-centre${query}`);
  return parseOrThrow(response, "Command centre unavailable");
}

export async function getBenchmarks() {
  const response = await fetch(`${API_BASE}/api/benchmarks`);
  return parseOrThrow(response, "Benchmarks unavailable");
}
