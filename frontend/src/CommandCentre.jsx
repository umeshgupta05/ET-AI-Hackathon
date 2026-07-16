/**
 * Command Centre — Threat Intelligence Dashboard
 *
 * Three panels:
 * 1. Interactive Geospatial Heatmap (Leaflet + OpenStreetMap)
 * 2. Interactive Fraud Network Graph (D3 force-directed)
 * 3. Predictive Threat Feed with trend indicators
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import * as d3 from "d3";
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Globe,
  Network,
  Shield,
  Activity,
  MapPin,
  BarChart3,
} from "lucide-react";
import "leaflet/dist/leaflet.css";
import { getThreatFeed, getCommandCentre, getBenchmarks, getGraphVisualization } from "./utils/api";

/* ── Severity color helper ── */
function severityColor(severity) {
  if (severity >= 0.8) return "#dc2626";
  if (severity >= 0.6) return "#ea580c";
  if (severity >= 0.4) return "#eab308";
  return "#16a34a";
}

function trendIcon(trend) {
  if (trend === "surging" || trend === "rising") return <TrendingUp size={16} />;
  if (trend === "declining") return <TrendingDown size={16} />;
  return <Minus size={16} />;
}

function trendClass(trend) {
  if (trend === "surging") return "trend-surging";
  if (trend === "rising") return "trend-rising";
  if (trend === "declining") return "trend-declining";
  return "trend-steady";
}

/* ── Force Graph Component ── */
function ForceGraph({ data }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = svgRef.current.clientWidth || 500;
    const height = 400;

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const nodes = (data.nodes || []).map((n, i) => ({
      ...n,
      id: n.id ?? i,
      x: width / 2 + Math.random() * 100 - 50,
      y: height / 2 + Math.random() * 100 - 50,
    }));

    const nodeMap = new Map(nodes.map((n, i) => [n.id, i]));

    const links = (data.edges || data.links || [])
      .filter((e) => nodeMap.has(e.source ?? e.from) && nodeMap.has(e.target ?? e.to))
      .map((e) => ({
        source: e.source ?? e.from,
        target: e.target ?? e.to,
        weight: e.weight ?? 1,
      }));

    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(12));

    const link = svg
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", (d) => Math.max(1, d.weight));

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => 4 + (d.risk_score || 0.3) * 10)
      .attr("fill", (d) => {
        if (d.label === "scammer" || d.risk_score > 0.7) return "#dc2626";
        if (d.label === "mule" || d.risk_score > 0.4) return "#ea580c";
        if (d.label === "victim") return "#eab308";
        return "#3b82f6";
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .attr("cursor", "pointer")
      .call(
        d3.drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    node.append("title").text((d) => `${d.label || d.id} — Risk: ${((d.risk_score || 0) * 100).toFixed(0)}%`);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });

    return () => simulation.stop();
  }, [data]);

  return <svg ref={svgRef} style={{ width: "100%", height: 400 }} />;
}

/* ── Main Component ── */
export default function CommandCentre() {
  const [threatFeed, setThreatFeed] = useState(null);
  const [commandData, setCommandData] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("map");

  useEffect(() => {
    async function loadAll() {
      setLoading(true);
      try {
        const [feed, cmd, bench, graph] = await Promise.allSettled([
          getThreatFeed(),
          getCommandCentre(),
          getBenchmarks(),
          getGraphVisualization(),
        ]);
        if (feed.status === "fulfilled") setThreatFeed(feed.value);
        if (cmd.status === "fulfilled") setCommandData(cmd.value);
        if (bench.status === "fulfilled") setBenchmarks(bench.value);
        if (graph.status === "fulfilled") setGraphData(graph.value);
      } catch (e) {
        console.error("Command centre load error:", e);
      }
      setLoading(false);
    }
    loadAll();

    // Auto-refresh threat feed every 10 seconds for live stats
    const interval = setInterval(async () => {
      try {
        const feed = await getThreatFeed();
        setThreatFeed(feed);
      } catch (e) { /* silent */ }
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="cc-loading">
        <Activity size={32} className="spin" />
        <p>Loading Threat Intelligence...</p>
      </div>
    );
  }

  const hotspots = commandData?.geospatial?.hotspots || [];
  const summary = threatFeed?.summary || {};

  return (
    <div className="command-centre">
      {/* ── Header Stats ── */}
      <div className="cc-stats-bar">
        <div className="cc-stat">
          <Activity size={18} className="stat-icon warning" />
          <div>
            <span className="stat-value">{summary.total_analyses || 0}</span>
            <span className="stat-label">Total Analyses</span>
          </div>
        </div>
        <div className="cc-stat">
          <BarChart3 size={18} className="stat-icon info" />
          <div>
            <span className="stat-value">{summary.analyses_24h || 0}</span>
            <span className="stat-label">Analyses (24h)</span>
          </div>
        </div>
        <div className="cc-stat">
          <AlertTriangle size={18} className="stat-icon critical" />
          <div>
            <span className="stat-value">{summary.threats_detected_24h || 0}</span>
            <span className="stat-label">Threats Detected</span>
          </div>
        </div>
        <div className="cc-stat">
          <Shield size={18} className="stat-icon success" />
          <div>
            <span className="stat-value">{summary.safe_cleared_24h || 0}</span>
            <span className="stat-label">Cleared Safe</span>
          </div>
        </div>
        <div className="cc-stat">
          <Network size={18} className="stat-icon danger" />
          <div>
            <span className="stat-value">{summary.active_patterns || 0}</span>
            <span className="stat-label">Scam Patterns</span>
          </div>
        </div>
        {threatFeed?.is_live && (
          <div className="cc-stat cc-live-indicator">
            <span className="live-dot" />
            <span className="stat-label">LIVE</span>
          </div>
        )}
      </div>

      {/* ── Tab Navigation ── */}
      <div className="cc-tabs">
        <button className={activeTab === "map" ? "cc-tab active" : "cc-tab"} onClick={() => setActiveTab("map")}>
          <Globe size={16} /> Geospatial Intelligence
        </button>
        <button className={activeTab === "graph" ? "cc-tab active" : "cc-tab"} onClick={() => setActiveTab("graph")}>
          <Network size={16} /> Fraud Network Graph
        </button>
        <button className={activeTab === "threats" ? "cc-tab active" : "cc-tab"} onClick={() => setActiveTab("threats")}>
          <AlertTriangle size={16} /> Threat Feed
        </button>
        <button className={activeTab === "metrics" ? "cc-tab active" : "cc-tab"} onClick={() => setActiveTab("metrics")}>
          <BarChart3 size={16} /> AI Benchmarks
        </button>
      </div>

      {/* ── Tab Content ── */}
      <div className="cc-content">
        {activeTab === "map" && (
          <div className="cc-panel">
            <h3>Crime Hotspot Intelligence — India</h3>
            <p className="cc-subtitle">
              {commandData?.geospatial?.source || "Reference intelligence feed"}. {commandData?.geospatial?.limitations}
            </p>
            <div className="cc-map-container">
              <MapContainer center={[22.5, 78.5]} zoom={5} style={{ height: 480, width: "100%", borderRadius: 8 }} scrollWheelZoom={true}>
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                />
                {hotspots.map((h, i) => (
                  <CircleMarker
                    key={i}
                    center={[h.lat, h.lon]}
                    radius={8 + h.reports * 0.6}
                    pathOptions={{
                      color: severityColor(h.severity),
                      fillColor: severityColor(h.severity),
                      fillOpacity: 0.6,
                      weight: 2,
                    }}
                  >
                    <Popup>
                      <div style={{ fontFamily: "Inter, sans-serif", fontSize: 13 }}>
                        <strong>{h.district}</strong>
                        <br />
                        Type: {h.type.replace(/_/g, " ")}
                        <br />
                        Reports: {h.reports}
                        <br />
                        Severity: {(h.severity * 100).toFixed(0)}%
                        <br />
                        Risk Score: {(h.risk_score * 100).toFixed(0)}%
                      </div>
                    </Popup>
                  </CircleMarker>
                ))}
              </MapContainer>
            </div>
            <div className="cc-legend">
              <span className="legend-item"><span className="legend-dot" style={{ background: "#dc2626" }} /> Critical (&gt;80%)</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#ea580c" }} /> High (60-80%)</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#eab308" }} /> Medium (40-60%)</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#16a34a" }} /> Low (&lt;40%)</span>
            </div>
          </div>
        )}

        {activeTab === "graph" && (
          <div className="cc-panel">
            <h3>Fraud Network Intelligence</h3>
            <p className="cc-subtitle">
              {commandData?.network?.total_nodes ?? 0} entities, {commandData?.network?.total_edges ?? 0} connections
            </p>
            <div className="cc-graph-container">
              {graphData ? (
                <ForceGraph data={graphData} />
              ) : (
                <div className="cc-empty">Graph data loading...</div>
              )}
            </div>
            <div className="cc-legend">
              <span className="legend-item"><span className="legend-dot" style={{ background: "#dc2626" }} /> Scammer (High Risk)</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#ea580c" }} /> Money Mule</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#eab308" }} /> Victim</span>
              <span className="legend-item"><span className="legend-dot" style={{ background: "#3b82f6" }} /> Entity (Bank/Phone)</span>
            </div>
          </div>
        )}

        {activeTab === "threats" && (
          <div className="cc-panel">
            <h3>Live Threat Intelligence Feed</h3>
            <p className="cc-subtitle">
              Real-time detections from system analyses — {summary.threats_detected_24h || 0} threats found,{" "}
              {summary.active_patterns || 0} distinct scam patterns identified.
            </p>
            {(threatFeed?.active_campaigns || []).length === 0 ? (
              <div className="cc-empty-state">
                <Shield size={40} />
                <h4>No threats detected yet</h4>
                <p>Run analyses from the Fraud Analysis tab — detected patterns will appear here in real-time.</p>
              </div>
            ) : (
              <div className="cc-threat-list">
                {(threatFeed?.active_campaigns || []).map((t, i) => (
                  <div key={i} className={`cc-threat-card ${trendClass(t.trend)}`}>
                    <div className="threat-header">
                      <div className="threat-title">
                        <span className="threat-severity-dot" style={{ background: severityColor(t.max_confidence || 0.5) }} />
                        {t.pattern}
                      </div>
                      <span className={`threat-trend ${trendClass(t.trend)}`}>
                        {trendIcon(t.trend)}
                        {t.trend}
                      </span>
                    </div>
                    <div className="threat-meta">
                      <span><Activity size={12} /> {t.count_24h} detections (24h)</span>
                      <span><BarChart3 size={12} /> {t.count_1h} in last hour</span>
                      <span><AlertTriangle size={12} /> Peak: {(t.max_confidence * 100).toFixed(0)}%</span>
                      <span><Shield size={12} /> Avg: {(t.avg_confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {threatFeed?.modality_breakdown && Object.keys(threatFeed.modality_breakdown).length > 0 && (
              <div className="cc-modality-bar">
                <h4>Analysis by Modality</h4>
                <div className="modality-chips">
                  {Object.entries(threatFeed.modality_breakdown).map(([mod, count]) => (
                    <span key={mod} className="modality-chip">{mod}: {count}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="cc-panel">
            <h3>AI Model Benchmarks</h3>
            <p className="cc-subtitle">
              {benchmarks?.disclosure || `Metrics recorded for ${benchmarks?.system_totals?.total_models ?? 0} locally evaluated models.`}
            </p>
            <div className="cc-metrics-grid">
              {(benchmarks?.models || []).map((m) => (
                <div key={m.name} className="cc-metric-card">
                  <h4>{m.name}</h4>
                  <p className="metric-type">{m.type}</p>
                  <div className="metric-bars">
                    {Object.entries(m.metrics || {}).map(([key, val]) => (
                      <div key={key} className="metric-row">
                        <span className="metric-key">{key.replace(/_/g, " ")}</span>
                        <div className="metric-bar-track">
                          <div
                            className="metric-bar-fill"
                            style={{
                              width: `${Math.min(val * 100, 100)}%`,
                              background: val >= 0.9 ? "#16a34a" : val >= 0.7 ? "#eab308" : "#ea580c",
                            }}
                          />
                        </div>
                        <span className="metric-val">{(val * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                  {m.evaluation_set && <p className="metric-eval">{m.evaluation_set}</p>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
