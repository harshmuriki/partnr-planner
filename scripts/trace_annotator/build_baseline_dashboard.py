"""Build a self-contained HTML dashboard comparing baseline runs; no simulation or model calls."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from export_run_workbook import SHEETS
from dashboard_subtasks import chart_data

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DEFAULT_ROWS = RESULTS / "run_comparison_20260929/baseline_runs_comparison.json"
FOLDERS = {
    "vlm_tamp_no_images": "image_comparison_20260927_232820/vlm_tamp_no_images",
    "vlm_tamp_images": "vlm_tamp_images_10min_20260928_111039",
    "react_no_images": "image_comparison_20260927_232820/react_no_images",
    "react_images": "image_comparison_20260927_232820/react_images",
    "tru_pomdp_no_images": "tru_pomdp_no_images_20260928_002709",
}
DIMENSIONS = {
    "task": ("Task", [1, 2, 3, 4, 5, 6, 7]),
    "memory": ("Memory", ["ACC", "INC", "OUT"]),
    "uncertainty": ("Uncertainty", ["BASE", "DIS", "ABS", "CON", "SUB"]),
}
ABLATIONS = [
    ("VLM-TAMP", "vlm_tamp_no_images", "vlm_tamp_images"),
    ("PARTNR ReAct", "react_no_images", "react_images"),
]
COST_KEYS = ("llm_planning_time_s", "llm_requests", "total_tokens", "llm_usd")


def load_costs(results=RESULTS):
    costs = {}
    for condition, folder in FOLDERS.items():
        for summary in sorted((results / folder).glob("task_*/experiment/summary.json")):
            for job in json.loads(summary.read_text())["jobs"]:
                cost = ((job.get("metrics") or {}).get("cost_metrics")) or {}
                costs[(condition, job["run_id"])] = {k: cost.get(k) for k in COST_KEYS}
    return costs


def mean_available(values):
    values = [v for v in values if v is not None]
    return (statistics.fmean(values) if values else None), len(values)


def summarize(rows):
    """Timeouts and unscored runs count as failures with zero progress."""
    n = len(rows)
    if not n:
        return None
    success = [r["simulator_outcome"] == "complete" for r in rows]
    completion = [r["completion"] or 0.0 for r in rows]
    solved = sum(r["subtasks_succeeded"] or 0 for r in rows)
    total = sum(r["num_subtasks"] or 0 for r in rows)
    runtimes = [r["runtime_s"] for r in rows if r["runtime_s"] is not None]
    by_iteration = defaultdict(list)
    for r, ok, c in zip(rows, success, completion):
        by_iteration[r["iteration"]].append((ok, c))
    iter_sr = [statistics.fmean(ok for ok, _ in v) for _, v in sorted(by_iteration.items())]
    iter_cr = [statistics.fmean(c for _, c in v) for _, v in sorted(by_iteration.items())]
    out = {
        "n": n,
        "successes": sum(success),
        "sr": sum(success) / n,
        "cr": statistics.fmean(completion),
        "sub_solved": solved,
        "sub_total": total,
        "sub_pct": solved / total if total else None,
        "sub_mean": solved / n,
        "timeouts": sum(r["execution_status"] == "timeout" for r in rows),
        "runtime_mean": statistics.fmean(runtimes) if runtimes else None,
        "runtime_median": statistics.median(runtimes) if runtimes else None,
        "runtime_success_mean": mean_available(
            [r["runtime_s"] for r, ok in zip(rows, success) if ok]
        )[0],
        "iter_sr_std": statistics.stdev(iter_sr) if len(iter_sr) > 1 else None,
        "iter_cr_std": statistics.stdev(iter_cr) if len(iter_cr) > 1 else None,
        "outcomes": dict(Counter(r["simulator_outcome"] for r in rows)),
    }
    out["timeout_rate"] = out["timeouts"] / n
    for key, field in (
        ("actions", "num_actions"),
        ("sim_steps", "sim_steps"),
        ("llm_time", "llm_planning_time_s"),
        ("llm_requests", "llm_requests"),
        ("tokens", "total_tokens"),
        ("usd", "llm_usd"),
    ):
        out[key + "_mean"], out[key + "_n"] = mean_available(r.get(field) for r in rows)
    return out


def run_record(r):
    return {
        "iteration": r["iteration"],
        "status": r["execution_status"],
        "outcome": r["simulator_outcome"],
        "solved": r["subtasks_succeeded"],
        "total": r["num_subtasks"],
        "completion": r["completion"],
        "runtime": r["runtime_s"],
        "actions": r["num_actions"],
        "sim_steps": r["sim_steps"],
        "usd": r.get("llm_usd"),
        "tokens": r.get("total_tokens"),
        "counts": r["counts"],
        "folder": r["run_folder"],
        "html_links": r.get("html_links", []),
        "folder_url": r.get("folder_url"),
    }


def build(rows, costs):
    rows = [dict(r, **costs.get((r["condition"], r["run_id"]), {})) for r in rows]
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    conditions = [c for c in SHEETS if c in by_condition]
    breakdowns = {}
    for dim, (label, values) in DIMENSIONS.items():
        breakdowns[dim] = {
            "label": label,
            "values": values,
            "stats": {
                str(v): {
                    c: summarize([r for r in by_condition[c] if r[dim] == v])
                    for c in conditions
                }
                for v in values
            },
        }
    variants = defaultdict(lambda: defaultdict(list))
    for r in rows:
        variants[(r["task"], r["task_id"], r["memory"], r["uncertainty"])][
            r["condition"]
        ].append(r)
    variant_rows = []
    for (task, task_id, memory, uncertainty), cells in sorted(variants.items()):
        variant_rows.append(
            {
                "task": task,
                "task_id": task_id,
                "memory": memory,
                "uncertainty": uncertainty,
                "cells": {
                    c: {
                        "successes": sum(r["simulator_outcome"] == "complete" for r in runs),
                        "n": len(runs),
                        "cr": statistics.fmean(r["completion"] or 0.0 for r in runs),
                        "runs": [run_record(r) for r in sorted(runs, key=lambda r: r["iteration"])],
                    }
                    for c, runs in cells.items()
                },
            }
        )
    return {
        "conditions": [{"key": c, "label": SHEETS[c]} for c in conditions],
        "overall": {c: summarize(by_condition[c]) for c in conditions},
        "breakdowns": breakdowns,
        "ablations": [
            {"label": label, "base": base, "images": images}
            for label, base, images in ABLATIONS
            if base in by_condition and images in by_condition
        ],
        "variants": variant_rows,
        "total_runs": len(rows),
        "subtasks": chart_data(rows),
    }


def render(data):
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    script = Path(__file__).with_name('dashboard_subtasks.js').read_text()
    return TEMPLATE.replace("__SUBTASK_SCRIPT__", script).replace("__DATA__", payload)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROWS.parent / "baseline_dashboard.html")
    args = parser.parse_args()
    data = build(json.loads(args.rows.read_text()), load_costs())
    args.output.write_text(render(data))
    for c in data["conditions"]:
        s = data["overall"][c["key"]]
        print(
            f"{c['label']:<24} n={s['n']} SR={s['sr']:.3f} CR={s['cr']:.3f} "
            f"subgoals={s['sub_solved']}/{s['sub_total']} timeouts={s['timeouts']} "
            f"outcomes={s['outcomes']}"
        )
    print(args.output.resolve())


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Final Baseline results dashboard</title>
<style>
:root{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#1d4ed8}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:20px 28px;background:#0f172a;color:#fff}
header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;color:#cbd5e1}
main{padding:20px 28px;max-width:1600px;margin:auto}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:20px}
h2{margin:0 0 12px;font-size:17px}
table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:#f1f5f9;font-weight:600;position:sticky;top:0;z-index:1}
th:first-child,td:first-child{text-align:left}
td.best{background:#dcfce7;font-weight:600}
.sub{color:var(--muted);font-size:12px}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:16px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px;border-top:4px solid}
.card .name{font-weight:600;margin-bottom:6px}
.card .kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.card .kpi b{display:block;font-size:20px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.chart h3{margin:0 0 4px;font-size:14px}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select{padding:4px 6px;font-size:13px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 10px;font-size:13px}
.scroll{overflow:auto;max-height:640px}
td.cell{cursor:pointer;text-align:center}
td.cell:hover{outline:2px solid var(--accent);outline-offset:-2px}
th.sortable{cursor:pointer}
.pos{color:#15803d;font-weight:600}.neg{color:#b91c1c;font-weight:600}
#detail{display:none}
.notes li{margin-bottom:4px}
svg text{font-size:11px;fill:#334155}
#subtaskTable td:first-child{white-space:normal;min-width:300px;max-width:480px}
#subtaskSection > .scroll{max-height:none}
#subtaskTable button{border:0;border-radius:4px;width:100%;min-width:220px;padding:8px;cursor:pointer;color:#0f172a;font:inherit;background:#f8fafc;text-align:left}
#subtaskTable button:hover,#subtaskTable button:focus{outline:2px solid var(--accent)}
#subtaskTable small{display:block;font-size:11px}
.subtask-bar{display:flex;height:24px;margin-top:6px;min-width:1px}
.subtask-bar span{display:block;height:100%}
.subtask-met{background:#15803d}.subtask-unmet{background:#dc2626}.subtask-unknown{background:#94a3b8}
#subtaskDetailTable td{white-space:normal;vertical-align:top;text-align:left}
#subtaskDetailTable td:nth-child(3){overflow-wrap:anywhere;max-width:550px}
#subtaskDetailTable pre{white-space:pre-wrap;overflow-wrap:anywhere}
#subtaskDetailTable td:first-child{min-width:160px}
#subtaskDetail{margin-top:18px}
</style>
</head>
<body>
<header>
<h1>Final Baseline results dashboard</h1>
<p id="subtitle"></p>
</header>
<main>
<section><h2>Overview</h2><div class="cards" id="cards"></div><div class="scroll"><table id="overview"></table></div></section>
<section><h2>Overall comparison</h2><div class="legend" id="legend1"></div><div class="charts" id="overallCharts"></div></section>
<section>
<h2>Breakdowns</h2>
<div class="controls">
<label>Split by <select id="dim"></select></label>
<label>Metric <select id="metric"></select></label>
</div>
<div class="legend" id="legend2"></div>
<div id="breakChart"></div>
<div class="scroll"><table id="breakTable"></table></div>
</section>
<section id="subtaskSection">
<h2>Subtask successes and failures by task</h2>
<p class="sub">Select a task to compare all its subtasks across VLM-TAMP images, ReAct images, and TRU-POMDP. Stacked bars show counts across the selected variants and all three repetitions, using the same count scale throughout the chart.</p>
<div class="controls">
<label>Task <select id="sTask"></select></label>
<label>Memory <select id="sMemory"><option value="">All memories</option></select></label>
<label>Uncertainty <select id="sUnc"><option value="">All variants</option></select></label>
<label><input type="checkbox" id="sFinalOnly"> Final scores only</label>
</div>
<div class="legend"><span><span class="swatch subtask-met"></span>Succeeded</span><span><span class="swatch subtask-unmet"></span>Failed</span><span><span class="swatch subtask-unknown"></span>Unknown</span></div>
<p class="sub" id="subtaskCoverage"></p>
<div class="scroll"><table id="subtaskTable"></table></div>
<p class="sub">Subtasks follow the ACC-BASE evaluation list; additional variant-specific criteria appear afterward. This is criterion order, not execution order. When final scores are missing, the latest saved evaluator state supplies checkpoint outcomes, labeled separately from final results. A checkpoint can predate termination. Use “Final scores only” to exclude these from succeeded/failed counts. Omitted criteria are not counted; unknown scores are not failures. Click a bar for checkpoint timing, repetitions and traces.</p>
<div id="subtaskDetail" hidden>
<h3 id="subtaskDetailTitle"></h3>
<div class="controls"><label>Show <select id="sStatus"><option value="unmet">Unmet</option><option value="unknown">Unknown / ambiguous</option><option value="met">Met</option><option value="">All outcomes</option></select></label></div>
<div class="scroll"><table id="subtaskDetailTable"></table></div>
</div>
</section>
<section hidden><h2>Image ablation (with images minus without images)</h2><div class="scroll"><table id="ablation"></table></div></section>
<section>
<h2>Per-variant results</h2>
<div class="controls">
<label>Task <select id="fTask"></select></label>
<label>Memory <select id="fMemory"></select></label>
<label>Uncertainty <select id="fUnc"></select></label>
<span class="sub">Cell: successes / iterations and mean completion. Click a cell for its runs; click a header to sort.</span>
</div>
<div class="scroll"><table id="variants"></table></div>
</section>
<section id="detail"><h2 id="detailTitle"></h2><div class="scroll"><table id="detailTable"></table></div></section>
<section class="notes"><h2>Metric definitions</h2><ul id="notes"></ul></section>
</main>
<script>
const DATA = __DATA__;
const BASELINE_COLORS = {vlm_tamp_images:"#16a34a", react_images:"#dc2626", tru_pomdp_no_images:"#eab308"};
const ENABLED = ["vlm_tamp_images","react_images","tru_pomdp_no_images"];
const C = ENABLED.map(key=>{const c=DATA.conditions.find(x=>x.key===key); return c?{...c,color:BASELINE_COLORS[key]}:null}).filter(Boolean);
const pct = x => x==null ? "–" : (100*x).toFixed(1)+"%";
const num = (x,d=1) => x==null ? "–" : Number(x).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
const secs = x => x==null ? "–" : num(x,1)+" s";
const usd = x => x==null ? "–" : "$"+Number(x).toFixed(4);
const esc = s => String(s).replace(/[&<>"]/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
const $ = id => document.getElementById(id);

const METRICS = {
  sr:{label:"Success rate (SR)",fmt:pct,better:"high",max:1},
  cr:{label:"Completion rate (CR)",fmt:pct,better:"high",max:1},
  sub_pct:{label:"Subgoals solved (%)",fmt:pct,better:"high",max:1},
  sub_mean:{label:"Subgoals solved per run",fmt:x=>num(x,2),better:"high"},
  timeout_rate:{label:"Timeout rate",fmt:pct,better:"low",max:1},
  runtime_mean:{label:"Mean runtime",fmt:secs,better:"low"},
  runtime_median:{label:"Median runtime",fmt:secs,better:"low"},
  runtime_success_mean:{label:"Mean runtime (successful runs)",fmt:secs,better:"low"},
  actions_mean:{label:"Mean actions",fmt:x=>num(x,1),better:"low"},
  sim_steps_mean:{label:"Mean sim steps",fmt:x=>num(x,0),better:"low"},
  llm_time_mean:{label:"Mean LLM planning time",fmt:secs,better:"low"},
  tokens_mean:{label:"Mean tokens",fmt:x=>num(x,0),better:"low"},
  usd_mean:{label:"Mean LLM cost",fmt:usd,better:"low"},
};

$("subtitle").textContent = DATA.subtitle || `${DATA.total_runs.toLocaleString()} runs across ${C.length} baselines · timeouts count as failures with zero progress`;
if(DATA.html_index){const p=document.createElement('p'),a=document.createElement('a');a.href=DATA.html_index;a.textContent='Browse HTML files by baseline and task';a.style.color='#93c5fd';p.appendChild(a);document.querySelector('header').appendChild(p);}

function legend(el){el.innerHTML=C.map(c=>`<span><span class="swatch" style="background:${c.color}"></span>${esc(c.label)}</span>`).join("");}
legend($("legend1")); legend($("legend2"));

function bestOf(values,better){
  const v=values.filter(x=>x!=null); if(!v.length) return null;
  return better==="high"?Math.max(...v):Math.min(...v);
}

// Cards
$("cards").innerHTML = C.map(c=>{const s=DATA.overall[c.key];return `<div class="card" style="border-top-color:${c.color}">
<div class="name">${esc(c.label)}</div><div class="kpis">
<div class="kpi"><b>${pct(s.sr)}</b><span class="sub">SR ± ${pct(s.iter_sr_std)}</span></div>
<div class="kpi"><b>${pct(s.cr)}</b><span class="sub">CR ± ${pct(s.iter_cr_std)}</span></div>
<div class="kpi"><b>${s.sub_solved}/${s.sub_total}</b><span class="sub">subgoals (${pct(s.sub_pct)})</span></div>
<div class="kpi"><b>${secs(s.runtime_mean)}</b><span class="sub">mean runtime</span></div>
<div class="kpi"><b>${pct(s.timeout_rate)}</b><span class="sub">timeouts (${s.timeouts})</span></div>
<div class="kpi"><b>${usd(s.usd_mean)}</b><span class="sub">mean cost (n=${s.usd_n})</span></div>
</div></div>`}).join("");

// Overview table
const OV = [
  ["n",{label:"Runs",fmt:x=>x}],["successes",{label:"Successes",fmt:x=>x,better:"high"}],
  ...["sr","cr","sub_pct","sub_mean"].map(k=>[k,METRICS[k]]),
  ["sub_solved",{label:"Subgoals solved",fmt:x=>x,better:"high"}],["sub_total",{label:"Subgoals total",fmt:x=>x}],
  ["timeouts",{label:"Timeouts",fmt:x=>x,better:"low"}],
  ...["timeout_rate","runtime_mean","runtime_median","runtime_success_mean","actions_mean","sim_steps_mean","llm_time_mean","tokens_mean","usd_mean"].map(k=>[k,METRICS[k]]),
];
const COVERAGE = {actions_mean:"actions_n",sim_steps_mean:"sim_steps_n",llm_time_mean:"llm_time_n",tokens_mean:"tokens_n",usd_mean:"usd_n"};
(function(){
  let h=`<tr><th>Metric</th>${C.map(c=>`<th><span class="swatch" style="background:${c.color}"></span>${esc(c.label)}</th>`).join("")}</tr>`;
  for(const [k,m] of OV){
    const vals=C.map(c=>DATA.overall[c.key][k]); const best=m.better?bestOf(vals,m.better):null;
    h+=`<tr><td>${esc(m.label)}${COVERAGE[k]?' <span class="sub">(available runs)</span>':''}</td>${C.map((c,i)=>{
      const cov=COVERAGE[k]?` <span class="sub">n=${DATA.overall[c.key][COVERAGE[k]]}</span>`:"";
      return `<td class="${best!=null&&vals[i]===best?'best':''}">${m.fmt(vals[i])}${cov}</td>`}).join("")}</tr>`;
  }
  $("overview").innerHTML=h;
})();

// Grouped bar chart
function barChart(groups, seriesFn, metric, opts={}){
  const m=METRICS[metric]; const W=opts.width||Math.max(360,groups.length*C.length*22+80), H=opts.height||230;
  const pad={l:52,r:10,t:18,b:34}; const iw=W-pad.l-pad.r, ih=H-pad.t-pad.b;
  const all=groups.flatMap(g=>C.map(c=>seriesFn(g,c))).filter(v=>v!=null);
  const max=m.max||Math.max(1e-9,...all)*1.1;
  const gw=iw/groups.length, bw=Math.min(28,(gw-10)/C.length);
  let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  for(let i=0;i<=4;i++){const y=pad.t+ih-ih*i/4, v=max*i/4;
    s+=`<line x1="${pad.l}" x2="${W-pad.r}" y1="${y}" y2="${y}" stroke="#e2e8f0"/><text x="${pad.l-6}" y="${y+4}" text-anchor="end">${m.max?Math.round(v*100)+"%":num(v,0)}</text>`;}
  groups.forEach((g,gi)=>{
    const x0=pad.l+gi*gw+(gw-bw*C.length)/2;
    C.forEach((c,ci)=>{const v=seriesFn(g,c); if(v==null) return; const h=ih*v/max, x=x0+ci*bw, y=pad.t+ih-h;
      s+=`<rect x="${x}" y="${y}" width="${bw-2}" height="${h}" fill="${c.color}"><title>${esc(c.label)} · ${esc(g.label)}: ${m.fmt(v)}</title></rect>`;
      if(opts.values) s+=`<text x="${x+(bw-2)/2}" y="${y-4}" text-anchor="middle">${m.fmt(v)}</text>`;});
    if(g.label) s+=`<text x="${pad.l+gi*gw+gw/2}" y="${H-12}" text-anchor="middle">${esc(g.label)}</text>`;
  });
  return s+"</svg>";
}

$("overallCharts").innerHTML=["sr","cr","sub_pct","runtime_mean","timeout_rate","usd_mean"].map(k=>
  `<div class="chart"><h3>${esc(METRICS[k].label)}</h3>${barChart([{label:""}],(g,c)=>DATA.overall[c.key][k],k,{width:360,values:true})}</div>`).join("");

// Breakdowns
const BM=["sr","cr","sub_pct","sub_mean","runtime_mean","runtime_median","timeout_rate","actions_mean","usd_mean"];
$("dim").innerHTML=Object.entries(DATA.breakdowns).map(([k,v])=>`<option value="${k}">${esc(v.label)}</option>`).join("");
$("metric").innerHTML=BM.map(k=>`<option value="${k}">${esc(METRICS[k].label)}</option>`).join("");
function renderBreakdown(){
  const dim=$("dim").value, metric=$("metric").value, b=DATA.breakdowns[dim], m=METRICS[metric];
  const groups=b.values.map(v=>({key:String(v),label:(dim==="task"?"Task ":"")+v}));
  $("breakChart").innerHTML=barChart(groups,(g,c)=>{const s=b.stats[g.key][c.key];return s?s[metric]:null},metric,{height:260});
  let h=`<tr><th>${esc(b.label)}</th>${C.map(c=>`<th><span class="swatch" style="background:${c.color}"></span>${esc(c.label)}</th>`).join("")}</tr>`;
  for(const g of groups){
    const vals=C.map(c=>{const s=b.stats[g.key][c.key];return s?s[metric]:null}); const best=bestOf(vals,m.better);
    h+=`<tr><td>${esc(g.label)} <span class="sub">n=${b.stats[g.key][C[0].key]?.n??0}/baseline</span></td>${vals.map(v=>`<td class="${v!=null&&v===best?'best':''}">${m.fmt(v)}</td>`).join("")}</tr>`;
  }
  $("breakTable").innerHTML=h;
}
$("dim").onchange=$("metric").onchange=renderBreakdown; renderBreakdown();

// Image ablation
(function(){
  const AM=["sr","cr","sub_pct","runtime_mean","timeout_rate"];
  const delta=(a,b,k)=>(a&&b&&a[k]!=null&&b[k]!=null)?b[k]-a[k]:null;
  const fmtD=(d,k)=>{if(d==null)return"–";const m=METRICS[k];const good=m.better==="high"?d>0:d<0;
    const txt=m.max?(d>0?"+":"")+(100*d).toFixed(1)+" pp":(d>0?"+":"")+num(d,1)+" s";
    return `<span class="${Math.abs(d)<1e-9?'':good?'pos':'neg'}">${txt}</span>`;};
  let h=`<tr><th rowspan="2">Slice</th>${DATA.ablations.map(a=>`<th colspan="${AM.length}" style="text-align:center">${esc(a.label)}</th>`).join("")}</tr>`;
  h+=`<tr>${DATA.ablations.map(()=>AM.map(k=>`<th>Δ ${esc(METRICS[k].label)}</th>`).join("")).join("")}</tr>`;
  const slices=[["Overall",DATA.overall]];
  for(const [dim,b] of Object.entries(DATA.breakdowns)) for(const v of b.values) slices.push([`${b.label} ${v}`,b.stats[String(v)]]);
  for(const [label,stats] of slices)
    h+=`<tr><td>${esc(label)}</td>${DATA.ablations.map(a=>AM.map(k=>`<td>${fmtD(delta(stats[a.base],stats[a.images],k),k)}</td>`).join("")).join("")}</tr>`;
  $("ablation").innerHTML=h;
})();

// Per-variant table
function opts(el,values){el.innerHTML=`<option value="">All</option>`+values.map(v=>`<option>${v}</option>`).join("");}
opts($("fTask"),DATA.breakdowns.task.values); opts($("fMemory"),DATA.breakdowns.memory.values); opts($("fUnc"),DATA.breakdowns.uncertainty.values);
let sortKey="task_id", sortDir=1;
function cellColor(cr){const r=Math.round(254-(254-187)*cr), g=Math.round(202+(247-202)*cr), b=Math.round(202+(208-202)*cr);return `rgb(${r},${g},${b})`;}
function renderVariants(){
  const ft=$("fTask").value, fm=$("fMemory").value, fu=$("fUnc").value;
  let rows=DATA.variants.filter(v=>(!ft||String(v.task)===ft)&&(!fm||v.memory===fm)&&(!fu||v.uncertainty===fu));
  const val=(v,k)=>k in v?v[k]:(v.cells[k]?v.cells[k].successes+v.cells[k].cr:-1);
  rows=[...rows].sort((a,b)=>{const x=val(a,sortKey),y=val(b,sortKey);return (x<y?-1:x>y?1:0)*sortDir;});
  const arrow=k=>sortKey===k?(sortDir>0?" ▲":" ▼"):"";
  let h=`<tr>${[["task_id","Variant"],["memory","Memory"],["uncertainty","Uncertainty"]].map(([k,l])=>`<th class="sortable" data-k="${k}">${l}${arrow(k)}</th>`).join("")}${C.map(c=>`<th class="sortable" data-k="${c.key}"><span class="swatch" style="background:${c.color}"></span>${esc(c.label)}${arrow(c.key)}</th>`).join("")}</tr>`;
  const idx=new Map(DATA.variants.map((v,i)=>[v,i]));
  for(const v of rows){
    h+=`<tr><td>${esc(v.task_id)}</td><td>${esc(v.memory)}</td><td>${esc(v.uncertainty)}</td>${C.map(c=>{const x=v.cells[c.key];
      return x?`<td class="cell" data-v="${idx.get(v)}" data-c="${c.key}" style="background:${cellColor(x.cr)}">${x.successes}/${x.n} · ${pct(x.cr)}</td>`:`<td>–</td>`}).join("")}</tr>`;
  }
  h+=`<tr><td colspan="3"><b>Shown: ${rows.length} variants</b></td>${C.map(c=>{const cells=rows.map(v=>v.cells[c.key]).filter(Boolean);
    const s=cells.reduce((a,x)=>a+x.successes,0), n=cells.reduce((a,x)=>a+x.n,0), cr=cells.reduce((a,x)=>a+x.cr*x.n,0);
    return `<td style="text-align:center"><b>${s}/${n} · ${pct(n?cr/n:null)}</b></td>`}).join("")}</tr>`;
  $("variants").innerHTML=h;
}
$("variants").onclick=e=>{
  const th=e.target.closest("th.sortable");
  if(th){const k=th.dataset.k; sortDir=sortKey===k?-sortDir:(k in METRICS||DATA.conditions.some(c=>c.key===k)?-1:1); sortKey=k; renderVariants(); return;}
  const td=e.target.closest("td.cell"); if(!td) return;
  const v=DATA.variants[+td.dataset.v], c=C.find(c=>c.key===td.dataset.c), cell=v.cells[c.key];
  $("detail").style.display="block";
  $("detailTitle").innerHTML=`<span class="swatch" style="background:${c.color}"></span>${esc(c.label)} · ${esc(v.task_id)}`;
  let h=`<tr><th>Iteration</th><th>Execution</th><th>Outcome</th><th>Subgoals</th><th>Completion</th><th>Runtime</th><th>Actions</th><th>Sim steps</th><th>Tokens</th><th>LLM cost</th><th>Action counts</th><th>Run folder</th></tr>`;
  for(const r of cell.runs) h+=`<tr><td>${r.iteration}</td><td>${esc(r.status)}</td><td>${esc(r.outcome)}</td><td>${r.solved??"–"}/${r.total}</td><td>${pct(r.completion)}</td><td>${secs(r.runtime)}</td><td>${r.actions??"–"}</td><td>${r.sim_steps??"–"}</td><td>${r.tokens==null?"–":num(r.tokens,0)}</td><td>${usd(r.usd)}</td>
    <td style="text-align:left">${esc(Object.entries(r.counts).map(([k,n])=>k+" "+n).join(", "))}</td><td style="text-align:left" class="sub">${r.folder_url?`<a href="${esc(r.folder_url)}" target="_blank" rel="noopener">Run files</a>`:esc(r.folder)}${(r.html_links||[]).map(l=>`<br><a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join('')}${r.folder_url&&!r.html_links.length?'<br>No saved HTML':''}</td></tr>`;
  $("detailTable").innerHTML=h;
  $("detail").scrollIntoView({behavior:"smooth"});
};
["fTask","fMemory","fUnc"].forEach(id=>$(id).onchange=renderVariants); renderVariants();

__SUBTASK_SCRIPT__

$("notes").innerHTML=(DATA.notes || [
  "Every baseline is scored over all of its runs. Timed-out or unscored runs count as failures: not successful, 0% completion, and 0 subgoals solved.",
  "SR: share of runs where the simulator reports the task as complete. ± is the standard deviation of SR across the three iterations.",
  "CR: mean simulator task_percent_complete. ± is the standard deviation of CR across the three iterations.",
  "Subgoals: simulator-scored propositions. Solved / total sums over runs; unscored prose from the task spec is excluded.",
  "Runtime: measured process wall time, including setup, exploration, planning, execution, and shutdown. Wall-clock limit: 600 s (10 min) for Tasks 1–5; 1800 s (30 min) for Tasks 6 and 7.",
  "Task 7 scoring: both toys inside the same wardrobe (any of four) and the plate inside any cabinet count as met; runs were rescored offline from saved final object positions.",
  "Actions, sim steps, tokens, and LLM cost average only the runs where they were saved; n shows how many. Timed-out runs have no saved cost metrics.",
  "Actions use the spreadsheet's convention: each Rearrange counts as 2 Navigate + 1 Pick + 1 Place, and Done is excluded.",
]).map(t=>`<li>${esc(t)}</li>`).join("");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
