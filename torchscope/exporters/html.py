"""
HTMLExporter — self-contained interactive HTML report via Plotly + Jinja2.

No server required. The output HTML embeds Plotly from CDN and inlines
all chart data as JSON. Works offline once loaded.

Charts included:
  1. GPU Utilization timeline  (line, per GPU)
  2. Memory waterfall          (area: allocated vs reserved)
  3. Temperature & Power       (dual-axis line)
  4. Top CUDA Kernels          (horizontal bar)
  5. Custom stage breakdown    (stacked bar) — only if stage data provided
  6. Findings panel            (severity-coloured cards)
  7. Recommendations list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    from jinja2 import Template
    _JINJA = True
except ImportError:
    _JINJA = False

# ── colour palette ────────────────────────────────────────────────────────────

_GPU_COLORS  = ["#4A90D9", "#27AE60", "#E67E22", "#9B59B6", "#E74C3C",
                "#1ABC9C", "#F39C12", "#2ECC71"]
_STAGE_PAL   = ["#5B9BD5", "#ED7D31", "#A9D18E", "#9E6FCE",
                "#F4C842", "#E74C3C", "#1ABC9C", "#BDC3C7"]
_SEV_BG      = {"HIGH": "#fdf0f0", "MEDIUM": "#fef9ec", "LOW": "#f0fdf4"}
_SEV_BORDER  = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
_SEV_BADGE   = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}


def _jsfig(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ── chart builders ────────────────────────────────────────────────────────────

def _chart_gpu_util(samples: list[dict]) -> Optional[str]:
    if not samples or not _PLOTLY:
        return None
    gpu_ids = sorted(set(s["gpu_id"] for s in samples))
    fig = go.Figure()
    for i, gid in enumerate(gpu_ids):
        pts = [s for s in samples if s["gpu_id"] == gid]
        utils = [s["gpu_util"] for s in pts]
        if all(v < 0 for v in utils):
            continue
        t0 = pts[0]["ts"]
        hex_color = _GPU_COLORS[i % len(_GPU_COLORS)]
        r, g, b   = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        fill_rgba = f"rgba({r},{g},{b},0.08)"
        fig.add_trace(go.Scatter(
            x=[s["ts"] - t0 for s in pts], y=utils,
            name=f"GPU {gid}", mode="lines",
            line=dict(color=hex_color, width=2),
            fill="tozeroy", fillcolor=fill_rgba,
        ))
    fig.add_hline(y=50, line_dash="dash", line_color="#aaa",
                  annotation_text="50% threshold")
    fig.update_layout(title="GPU Utilization", xaxis_title="Elapsed (s)",
                      yaxis_title="Util %", yaxis_range=[0, 100],
                      template="plotly_white", height=300,
                      margin=dict(t=40, b=40))
    return _jsfig(fig)


def _chart_memory(snapshots: list[dict]) -> Optional[str]:
    if not snapshots or not _PLOTLY:
        return None
    t0 = snapshots[0]["ts"]
    xs = [s["ts"] - t0 for s in snapshots]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=[s["allocated_gb"] for s in snapshots],
        name="Allocated", fill="tozeroy", mode="lines",
        line=dict(color="#4A90D9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=[s["reserved_gb"] for s in snapshots],
        name="Reserved", mode="lines",
        line=dict(color="#ED7D31", width=1.5, dash="dash"),
    ))
    fig.update_layout(title="GPU Memory", xaxis_title="Elapsed (s)",
                      yaxis_title="GB", template="plotly_white",
                      height=300, margin=dict(t=40, b=40))
    return _jsfig(fig)


def _chart_temp_power(samples: list[dict]) -> Optional[str]:
    if not samples or not _PLOTLY:
        return None
    pts = [s for s in samples if s.get("temp_c", -1) >= 0]
    if not pts:
        return None
    t0 = pts[0]["ts"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=[s["ts"] - t0 for s in pts], y=[s["temp_c"] for s in pts],
        name="Temp (°C)", line=dict(color="#e74c3c", width=2),
    ), secondary_y=False)
    pwr_pts = [s for s in pts if s.get("power_w", -1) >= 0]
    if pwr_pts:
        fig.add_trace(go.Scatter(
            x=[s["ts"] - t0 for s in pwr_pts], y=[s["power_w"] for s in pwr_pts],
            name="Power (W)", line=dict(color="#f39c12", width=2),
        ), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color="#e74c3c",
                  annotation_text="80°C throttle", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Power (W)", secondary_y=True)
    fig.update_layout(title="Temperature & Power", xaxis_title="Elapsed (s)",
                      template="plotly_white", height=300, margin=dict(t=40, b=40))
    return _jsfig(fig)


def _chart_kernels(top_kernels: list[dict]) -> Optional[str]:
    if not top_kernels or not _PLOTLY:
        return None
    kerns = top_kernels[:15]
    names = [k["name"][:55] for k in kerns]
    times = [k["cuda_time_ms"] for k in kerns]
    fig = go.Figure(go.Bar(
        x=times, y=names, orientation="h",
        marker_color="#4A90D9",
        text=[f"{t:.2f} ms" for t in times],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top CUDA Kernels by Time",
        xaxis_title="CUDA time (ms)", template="plotly_white",
        height=max(320, len(kerns) * 30),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=40, b=40, l=300),
    )
    return _jsfig(fig)


def _chart_stages(stages: dict) -> Optional[str]:
    if not stages or not _PLOTLY:
        return None
    names  = list(stages.keys())
    values = [stages[n] * 1000 for n in names]  # → ms
    colors = [_STAGE_PAL[i % len(_STAGE_PAL)] for i in range(len(names))]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.1f} ms" for v in values],
        textposition="outside",
    ))
    total = sum(values)
    pcts  = [v / total * 100 for v in values] if total > 0 else [0] * len(values)
    fig.update_layout(
        title="Pipeline Stage Breakdown — Lower is Better",
        yaxis_title="Time (ms)", template="plotly_white",
        height=340, margin=dict(t=40, b=40),
        annotations=[
            dict(x=i, y=v + max(values) * 0.03,
                 text=f"{p:.1f}%", showarrow=False,
                 font=dict(size=10, color="#555"))
            for i, (v, p) in enumerate(zip(values, pcts))
        ],
    )
    return _jsfig(fig)


# ── HTML template ─────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>torchscope — {{ title }}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body   { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f4f6fb; color: #1a1a2e; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
         color: #fff; padding: 28px 48px; }
header h1  { font-size: 1.5rem; font-weight: 700; letter-spacing: -.3px; }
header .sub { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
.container  { max-width: 1400px; margin: 0 auto; padding: 36px 28px; }
.section    { margin-bottom: 40px; }
.section h2 { font-size: 1rem; font-weight: 700; color: #1a1a2e;
              border-left: 4px solid #4A90D9; padding-left: 12px;
              margin-bottom: 18px; text-transform: uppercase;
              letter-spacing: .05em; }
.grid-2     { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3     { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.card       { background: #fff; border-radius: 12px; padding: 20px;
              box-shadow: 0 2px 12px rgba(0,0,0,.07); }
.chart-card { background: #fff; border-radius: 12px; padding: 18px;
              box-shadow: 0 2px 12px rgba(0,0,0,.07); }

/* kpi row */
.kpi-row  { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 32px; }
.kpi      { background: #fff; border-radius: 12px; padding: 18px 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,.07); flex: 1; min-width: 130px; }
.kpi .val { font-size: 1.9rem; font-weight: 700; color: #1a1a2e; line-height: 1; }
.kpi .lbl { font-size: 0.73rem; color: #8892b0; margin-top: 5px;
            text-transform: uppercase; letter-spacing: .06em; }

/* findings */
.finding     { border-left: 4px solid; border-radius: 0 10px 10px 0;
               padding: 13px 18px; margin-bottom: 10px;
               box-shadow: 0 1px 5px rgba(0,0,0,.05); }
.badge       { display: inline-block; font-size: 0.68rem; font-weight: 700;
               padding: 2px 9px; border-radius: 4px; margin-bottom: 5px;
               color: #fff; text-transform: uppercase; letter-spacing: .07em; }
.finding .type-label { font-size: 0.78rem; color: #8892b0; margin-left: 8px; }
.finding .detail     { font-size: 0.9rem; margin-top: 2px; }

/* recommendations */
.rec-list    { list-style: none; }
.rec-list li { background: #fff; border-radius: 10px; padding: 13px 18px;
               margin-bottom: 8px; font-size: 0.9rem;
               box-shadow: 0 1px 5px rgba(0,0,0,.05);
               border-left: 3px solid #4A90D9; }
.rec-list li::before { content: "→ "; font-weight: 700; color: #4A90D9; }

/* meta table */
.meta-table  { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.meta-table td { padding: 7px 14px; border-bottom: 1px solid #eef0f6; }
.meta-table tr:last-child td { border-bottom: none; }
.meta-table td:first-child { color: #8892b0; width: 200px; }

footer { text-align: center; padding: 28px; font-size: 0.78rem; color: #aaa; }
@media (max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<header>
  <h1>torchscope</h1>
  <div class="sub">{{ title }} &nbsp;·&nbsp; {{ timestamp }}</div>
</header>

<div class="container">

  <!-- KPI row -->
  <div class="kpi-row">
    <div class="kpi">
      <div class="val" style="color:{% if summary.high_severity > 0 %}#e74c3c{% else %}#27ae60{% endif %}">
        {{ summary.high_severity }}
      </div>
      <div class="lbl">High severity</div>
    </div>
    <div class="kpi">
      <div class="val" style="color:#f39c12">{{ summary.medium_severity }}</div>
      <div class="lbl">Medium severity</div>
    </div>
    <div class="kpi">
      <div class="val">{{ gpu_kpis.avg_gpu_util }}</div>
      <div class="lbl">Avg GPU util %</div>
    </div>
    <div class="kpi">
      <div class="val">{{ gpu_kpis.peak_mem_gb }} GB</div>
      <div class="lbl">Peak memory</div>
    </div>
    <div class="kpi">
      <div class="val">{{ gpu_kpis.avg_power_w }} W</div>
      <div class="lbl">Avg power</div>
    </div>
    <div class="kpi">
      <div class="val">{{ gpu_kpis.duration_s }} s</div>
      <div class="lbl">Duration</div>
    </div>
  </div>

  <!-- Findings -->
  <div class="section">
    <h2>Findings ({{ summary.total_findings }})</h2>
    {% if findings %}
      {% for f in findings %}
      <div class="finding" style="border-color:{{ sev_border[f.severity] }};background:{{ sev_bg[f.severity] }}">
        <span class="badge" style="background:{{ sev_badge[f.severity] }}">{{ f.severity }}</span>
        <span class="type-label">{{ f.type }}</span>
        <div class="detail">{{ f.detail }}</div>
      </div>
      {% endfor %}
    {% else %}
      <p style="color:#27ae60;font-weight:600;padding:12px 0">No bottlenecks detected.</p>
    {% endif %}
  </div>

  <!-- Recommendations -->
  {% if recommendations %}
  <div class="section">
    <h2>Recommendations</h2>
    <ul class="rec-list">
      {% for r in recommendations %}<li>{{ r }}</li>{% endfor %}
    </ul>
  </div>
  {% endif %}

  <!-- GPU util + Memory -->
  {% if chart_gpu_util or chart_memory %}
  <div class="section">
    <h2>GPU Hardware Telemetry</h2>
    <div class="grid-2">
      {% if chart_gpu_util %}<div class="chart-card" id="c-gpu-util"></div>{% endif %}
      {% if chart_memory   %}<div class="chart-card" id="c-memory"></div>{% endif %}
    </div>
  </div>
  {% endif %}

  <!-- Temperature & Power -->
  {% if chart_temp_power %}
  <div class="section">
    <h2>Temperature & Power</h2>
    <div class="chart-card" id="c-temp-power"></div>
  </div>
  {% endif %}

  <!-- Kernel breakdown -->
  {% if chart_kernels %}
  <div class="section">
    <h2>Top CUDA Kernels</h2>
    <div class="chart-card" id="c-kernels"></div>
  </div>
  {% endif %}

  <!-- Stage breakdown -->
  {% if chart_stages %}
  <div class="section">
    <h2>Pipeline Stage Breakdown</h2>
    <div class="chart-card" id="c-stages"></div>
  </div>
  {% endif %}

  <!-- Run metadata -->
  <div class="section">
    <h2>Run Metadata</h2>
    <div class="card">
      <table class="meta-table">
        {% for k, v in metadata.items() %}
        <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>

</div>
<footer>generated by <strong>torchscope {{ version }}</strong></footer>

<script>
function render(id, data) {
  if (!data || !document.getElementById(id)) return;
  var fig = JSON.parse(data);
  Plotly.newPlot(id, fig.data, fig.layout,
    {responsive: true, displayModeBar: false});
}
render("c-gpu-util",   {{ chart_gpu_util   | tojson }});
render("c-memory",     {{ chart_memory     | tojson }});
render("c-temp-power", {{ chart_temp_power | tojson }});
render("c-kernels",    {{ chart_kernels    | tojson }});
render("c-stages",     {{ chart_stages     | tojson }});
</script>
</body>
</html>
"""


class HTMLExporter:
    """
    Generates a self-contained HTML profiling report.

    Usage:
        exp = HTMLExporter()
        exp.generate(
            output_path   = "report.html",
            title         = "BERT fine-tune — A100",
            analysis      = analyzer.analyze(),
            gpu_samples   = gpu_collector.samples,
            mem_snapshots = mem_collector.snapshots,
            top_kernels   = tracer.top_kernels(),
            custom_stages = {"decode": 1.5, "preprocess": 0.1, "inference": 0.9},
        )
    """

    def generate(
        self,
        output_path:   str,
        title:         str,
        analysis:      dict,
        gpu_samples:   list[dict]       = None,
        mem_snapshots: list[dict]       = None,
        top_kernels:   list[dict]       = None,
        custom_stages: dict             = None,
        metadata:      dict             = None,
    ) -> str:
        if not _JINJA:
            raise ImportError("pip install jinja2")
        if not _PLOTLY:
            raise ImportError("pip install plotly")

        from datetime import datetime
        from torchscope import __version__

        gpu_summary = analysis.get("_gpu_summary", {})
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ctx = dict(
            title          = title,
            timestamp      = ts,
            version        = __version__,
            summary        = analysis["summary"],
            findings       = analysis["findings"],
            recommendations= analysis["recommendations"],
            sev_bg         = _SEV_BG,
            sev_border     = _SEV_BORDER,
            sev_badge      = _SEV_BADGE,
            gpu_kpis       = {
                "avg_gpu_util": gpu_summary.get("avg_gpu_util", "N/A"),
                "peak_mem_gb":  gpu_summary.get("peak_mem_gb",  "N/A"),
                "avg_power_w":  gpu_summary.get("avg_power_w",  "N/A"),
                "duration_s":   gpu_summary.get("duration_s",   "N/A"),
            },
            metadata       = metadata or {},
            chart_gpu_util  = _chart_gpu_util(gpu_samples   or []),
            chart_memory    = _chart_memory(mem_snapshots   or []),
            chart_temp_power= _chart_temp_power(gpu_samples or []),
            chart_kernels   = _chart_kernels(top_kernels    or []),
            chart_stages    = _chart_stages(custom_stages   or {}),
        )

        html = Template(_TEMPLATE).render(**ctx)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"torchscope: report → {output_path}")
        return output_path
