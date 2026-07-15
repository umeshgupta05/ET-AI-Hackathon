# Digital Public Safety Shield Frontend

React 19 + Vite client for the Digital Public Safety Shield. The frontend is not standalone: start the FastAPI backend first by following the repository [root README](../README.md).

## Requirements

- Node.js 20.19+ or 22.12+
- Backend running at `http://localhost:8000`

## Development

```powershell
cd "D:\ET AI Hackathon\frontend"
npm install
npm run dev
```

Open http://localhost:5173.

The API defaults to `http://localhost:8000`. To use a different backend, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

Restart Vite after changing environment variables.

## Quality Checks

```powershell
npm run lint
npm run i18n:check
npm run build
```

Preview the production build locally:

```powershell
npm run preview
```

Vite prints the preview URL, normally `http://localhost:4173`.

## Implemented Flows

- Register, sign in, profile, session persistence, and case history
- Text, image, audio upload, and in-browser microphone recording
- Scam and legitimate transcript demos
- Turn-by-turn confidence trajectory
- Agent trace, model evidence, recommendations, and report guidance
- Fraud graph and geospatial hotspot views
- Evidence export and browser verdict playback
- Responsive interface with 113 checked strings across all 12 supported languages
- Request-time localized Kimi explanations, recommendations, and turn-level reasoning

## Important Notes

- Browser microphone access requires `localhost` or HTTPS and user permission.
- Hosted Kimi/Whisper features require backend API keys; keys never belong in frontend environment files.
- `dist/`, `node_modules/`, and `.env.local` are Git-ignored.
- `npm run i18n:generate` regenerates generated regional catalogs using the configured backend Groq key; normal builds and language switching do not call a translation service.
- Do not treat UI verdicts as legal, banking, or forensic certification; uncertain cases require human review.

## Backend Integrations

The browser continues to use synchronous REST/WebSocket APIs. Optional infrastructure stays behind the API, so broker, Redis, and MCP credentials never enter frontend code:

- RabbitMQ provides authenticated background text jobs and worker retries.
- Redis provides shared API/login rate limits; rejected requests include `429` and `Retry-After`.
- MCP provides a separate trusted analyst-assistant interface and is not exposed in the citizen UI.

See the root [README](../README.md#optional-infrastructure) for setup, security boundaries, and integration tests.

## Key Files

```text
src/App.jsx          Main application and workflows
src/index.css        Responsive design system
src/utils/api.js     REST/auth client and API base configuration
vite.config.js       Vite configuration
```
