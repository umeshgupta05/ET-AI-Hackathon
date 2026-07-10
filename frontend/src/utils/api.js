const API_BASE = "http://localhost:8000";

let accessToken = null;

export function setAccessToken(token) {
  accessToken = token || null;
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

export async function updateMe(payload) {
  const response = await fetch(`${API_BASE}/api/auth/me`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response, "Profile update failed");
}

export async function getHistory() {
  const response = await fetch(`${API_BASE}/api/history`, {
    headers: authHeaders(),
  });
  return parseOrThrow(response, "History load failed");
}

export async function analyzeMultimodal({ text, image, audio }) {
  const formData = new FormData();
  if (text) formData.append("text", text);
  if (image) formData.append("image", image);
  if (audio) formData.append("audio", audio);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });

  return parseOrThrow(response, "Analysis failed");
}

export async function analyzeText(text) {
  const response = await fetch(`${API_BASE}/api/analyze/text`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ text }),
  });
  return parseOrThrow(response, "Text analysis failed");
}

export async function analyzeTurnByTurn(turns) {
  const response = await fetch(`${API_BASE}/api/analyze/turns`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ turns }),
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
