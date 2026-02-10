#!/usr/bin/env python3
"""
Web GUI for browsing and annotating experiment runs.

Usage:
    python scripts/run_browser_app.py [--results-dir PATH] [--port PORT]
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, render_template_string, request, send_file

app = Flask(__name__)

# Global config
RESULTS_DIR = None
ANNOTATIONS_FILE = None


def load_annotations() -> Dict:
    """Load notes and stars from JSON file."""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, "r") as f:
            return json.load(f)
    return {"runs": {}}


def save_annotations(data: Dict) -> None:
    """Save notes and stars to JSON file."""
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def scan_runs_directory() -> List[Dict]:
    """Scan the results directory for all run folders."""
    runs = []

    if not RESULTS_DIR.exists():
        return runs

    annotations = load_annotations()

    for item in RESULTS_DIR.iterdir():
        if not item.is_dir():
            continue

        # Get basic info
        run_info = {
            "name": item.name,
            "path": str(item.relative_to(RESULTS_DIR)),
            "mtime": item.stat().st_mtime,
            "mtime_str": datetime.fromtimestamp(item.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        # Count episodes and HTML files
        episode_dirs = [d for d in item.iterdir() if d.is_dir()]
        run_info["episode_count"] = len(episode_dirs)
        run_info["episodes"] = [f"{item.name}/{d.name}" for d in episode_dirs]

        html_files = list(item.rglob("*.html"))
        run_info["html_count"] = len(html_files)
        
        # Count video files
        video_files = list(item.rglob("*.mp4"))
        run_info["video_count"] = len(video_files)
        run_info["has_videos"] = len(video_files) > 0

        # Add annotations
        run_key = run_info["path"]
        if run_key in annotations["runs"]:
            run_info["starred"] = annotations["runs"][run_key].get("starred", False)
            run_info["notes"] = annotations["runs"][run_key].get("notes", "")
        else:
            run_info["starred"] = False
            run_info["notes"] = ""

        runs.append(run_info)

    # Sort by modification time, most recent first
    runs.sort(key=lambda x: x["mtime"], reverse=True)

    return runs


def get_run_files(run_path: str) -> Dict:
    """Get all files in a run, organized by episode."""
    full_path = RESULTS_DIR / run_path

    if not full_path.exists():
        return {"episodes": []}

    episodes = []

    for episode_dir in sorted(full_path.iterdir()):
        if not episode_dir.is_dir():
            continue

        episode_info = {
            "name": f"{run_path}/{episode_dir.name}",
            "html_files": [],
            "video_files": [],
            "other_files": [],
        }

        # Find HTML files
        for html_file in episode_dir.rglob("*.html"):
            rel_path = str(html_file.relative_to(RESULTS_DIR))
            episode_info["html_files"].append({
                "name": html_file.name,
                "path": rel_path,
                "dir": str(html_file.parent.relative_to(episode_dir)),
            })

        # Find video files
        for ext in ["*.mp4", "*.avi", "*.webm"]:
            for video_file in episode_dir.rglob(ext):
                rel_path = str(video_file.relative_to(RESULTS_DIR))
                episode_info["video_files"].append({
                    "name": video_file.name,
                    "path": rel_path,
                })

        # Find trace/log files
        for ext in ["*.txt", "*.json", "*.gz"]:
            for file in episode_dir.rglob(ext):
                if file.suffix == ".html":
                    continue
                rel_path = str(file.relative_to(RESULTS_DIR))
                episode_info["other_files"].append({
                    "name": file.name,
                    "path": rel_path,
                    "size": file.stat().st_size,
                })

        episodes.append(episode_info)

    return {"episodes": episodes}


# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Run Browser</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 { margin: 0; font-size: 1.5rem; }
        .header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.9rem; }
        .container {
            display: flex;
            height: calc(100vh - 80px);
        }
        .sidebar {
            width: 400px;
            background: white;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            padding: 1rem;
        }
        .content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 2rem;
        }
        .run-card {
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            background: white;
        }
        .run-card:hover {
            border-color: #3498db;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .run-card.active {
            border-color: #3498db;
            background: #e3f2fd;
        }
        .run-card.starred {
            border-color: #f39c12;
            background: #fff8e1;
        }
        .run-card.no-videos {
            opacity: 0.5;
            background: #ffebee;
            border-color: #ef5350;
        }
        .run-card.no-videos .run-name {
            color: #c62828;
        }
        .run-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .run-name {
            font-weight: 600;
            font-size: 0.9rem;
            color: #2c3e50;
        }
        .star-btn {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        .star-btn.starred { color: #f39c12; }
        .star-btn:not(.starred) { color: #ccc; }
        .run-meta {
            font-size: 0.75rem;
            color: #7f8c8d;
            margin-bottom: 0.5rem;
        }
        .run-notes {
            font-size: 0.8rem;
            color: #555;
            font-style: italic;
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .episode-section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .episode-section h3 {
            margin: 0 0 1rem 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .file-group {
            margin-bottom: 1.5rem;
        }
        .file-group h4 {
            margin: 0 0 0.5rem 0;
            color: #34495e;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .file-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .file-item {
            padding: 0.5rem;
            margin-bottom: 0.25rem;
            background: #f8f9fa;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .file-item a {
            color: #3498db;
            text-decoration: none;
            font-size: 0.9rem;
        }
        .file-item a:hover {
            text-decoration: underline;
        }
        .file-path {
            font-size: 0.75rem;
            color: #95a5a6;
            margin-left: 0.5rem;
        }
        .notes-section {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .notes-section h3 {
            margin: 0 0 1rem 0;
            color: #2c3e50;
        }
        .notes-section textarea {
            width: 100%;
            min-height: 120px;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
            font-size: 0.9rem;
            resize: vertical;
        }
        .notes-section button {
            margin-top: 0.5rem;
            padding: 0.5rem 1rem;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .notes-section button:hover {
            background: #2980b9;
        }
        .empty-state {
            text-align: center;
            color: #95a5a6;
            padding: 3rem;
        }
        .filter-section {
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .filter-section input {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            background: #ecf0f1;
            border-radius: 12px;
            font-size: 0.7rem;
            margin-right: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Experiment Run Browser</h1>
        <p id="results-path">Loading...</p>
    </div>
    <div class="container">
        <div class="sidebar">
            <div class="filter-section">
                <input type="text" id="filter-input" placeholder="Filter runs...">
            </div>
            <div id="runs-list">Loading...</div>
        </div>
        <div class="content">
            <div class="empty-state">
                <h2>👈 Select a run to view details</h2>
                <p>Click on a run in the sidebar to see its HTML files, traces, and videos.</p>
            </div>
        </div>
    </div>

    <script>
        let currentRun = null;
        let allRuns = [];

        async function loadRuns() {
            const response = await fetch('/api/runs');
            const data = await response.json();
            allRuns = data.runs;
            document.getElementById('results-path').textContent = data.results_dir;
            renderRuns(allRuns);
        }

        function renderRuns(runs) {
            const container = document.getElementById('runs-list');
            if (runs.length === 0) {
                container.innerHTML = '<div class="empty-state"><p>No runs found</p></div>';
                return;
            }
            container.innerHTML = runs.map(run => `
                <div class="run-card ${run.starred ? 'starred' : ''} ${!run.has_videos ? 'no-videos' : ''} ${currentRun === run.path ? 'active' : ''}" 
                     onclick="selectRun('${run.path}')">
                    <div class="run-header">
                        <div class="run-name">${run.name}</div>
                        <button class="star-btn ${run.starred ? 'starred' : ''}" 
                                onclick="toggleStar(event, '${run.path}')">★</button>
                    </div>
                    <div class="run-meta">
                        <span class="badge">${run.episode_count} episodes</span>
                        <span class="badge">${run.html_count} HTML files</span>
                        <br>${run.mtime_str}
                    </div>
                    ${run.episodes && run.episodes.length > 0 ? `<div class="run-notes" style="font-style: normal; font-size: 0.75rem;">${run.episodes.join(', ')}</div>` : ''}
                    ${run.notes ? `<div class="run-notes">${run.notes.substring(0, 100)}${run.notes.length > 100 ? '...' : ''}</div>` : ''}
                </div>
            `).join('');
        }

        async function selectRun(path) {
            currentRun = path;
            renderRuns(allRuns);
            
            const response = await fetch(`/api/run/${encodeURIComponent(path)}`);
            const data = await response.json();
            
            const content = document.querySelector('.content');
            
            if (data.episodes.length === 0) {
                content.innerHTML = '<div class="empty-state"><p>No files found in this run</p></div>';
                return;
            }

            const run = allRuns.find(r => r.path === path);
            
            content.innerHTML = `
                <div class="notes-section">
                    <h3>📝 Notes</h3>
                    <textarea id="notes-textarea">${run.notes || ''}</textarea>
                    <button onclick="saveNotes()">Save Notes</button>
                </div>
                ${data.episodes.map(episode => `
                    <div class="episode-section">
                        <h3>${episode.name}</h3>
                        ${episode.html_files.length > 0 ? `
                            <div class="file-group">
                                <h4>📄 HTML Files</h4>
                                <ul class="file-list">
                                    ${episode.html_files.map(file => `
                                        <li class="file-item">
                                            <div>
                                                <a href="/file/${encodeURIComponent(file.path)}" target="_blank">${file.name}</a>
                                                <span class="file-path">${file.dir}</span>
                                            </div>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        ${episode.video_files.length > 0 ? `
                            <div class="file-group">
                                <h4>🎥 Videos</h4>
                                <ul class="file-list">
                                    ${episode.video_files.map(file => `
                                        <li class="file-item">
                                            <a href="/file/${encodeURIComponent(file.path)}" target="_blank">${file.name}</a>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        ${episode.other_files.length > 0 ? `
                            <div class="file-group">
                                <h4>📦 Other Files</h4>
                                <ul class="file-list">
                                    ${episode.other_files.map(file => `
                                        <li class="file-item">
                                            <div>
                                                <a href="/file/${encodeURIComponent(file.path)}" target="_blank">${file.name}</a>
                                                <span class="file-path">${(file.size / 1024).toFixed(1)} KB</span>
                                            </div>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            `;
        }

        async function toggleStar(event, path) {
            event.stopPropagation();
            await fetch(`/api/star/${encodeURIComponent(path)}`, { method: 'POST' });
            await loadRuns();
        }

        async function saveNotes() {
            const notes = document.getElementById('notes-textarea').value;
            await fetch(`/api/notes/${encodeURIComponent(currentRun)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ notes })
            });
            await loadRuns();
            alert('Notes saved!');
        }

        document.getElementById('filter-input').addEventListener('input', (e) => {
            const filter = e.target.value.toLowerCase();
            const filtered = allRuns.filter(run => 
                run.name.toLowerCase().includes(filter) || 
                (run.notes && run.notes.toLowerCase().includes(filter))
            );
            renderRuns(filtered);
        });

        loadRuns();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/runs")
def api_runs():
    runs = scan_runs_directory()
    return jsonify({"runs": runs, "results_dir": str(RESULTS_DIR)})


@app.route("/api/run/<path:run_path>")
def api_run_files(run_path):
    return jsonify(get_run_files(run_path))


@app.route("/api/star/<path:run_path>", methods=["POST"])
def api_star(run_path):
    annotations = load_annotations()
    if run_path not in annotations["runs"]:
        annotations["runs"][run_path] = {"starred": False, "notes": ""}
    annotations["runs"][run_path]["starred"] = not annotations["runs"][run_path][
        "starred"
    ]
    save_annotations(annotations)
    return jsonify({"success": True})


@app.route("/api/notes/<path:run_path>", methods=["POST"])
def api_notes(run_path):
    annotations = load_annotations()
    if run_path not in annotations["runs"]:
        annotations["runs"][run_path] = {"starred": False, "notes": ""}
    annotations["runs"][run_path]["notes"] = request.json.get("notes", "")
    save_annotations(annotations)
    return jsonify({"success": True})


@app.route("/file/<path:file_path>")
def serve_file(file_path):
    full_path = RESULTS_DIR / file_path
    if not full_path.exists():
        return "File not found", 404
    return send_file(full_path)


def main():
    global RESULTS_DIR, ANNOTATIONS_FILE

    parser = argparse.ArgumentParser(
        description="Web GUI for browsing experiment runs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/single_agent_zero_shot_react_summary",
        help="Path to results directory",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run the server on",
    )
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir).resolve()
    ANNOTATIONS_FILE = RESULTS_DIR / ".annotations.json"

    print(f"\n🚀 Starting Run Browser...")
    print(f"📂 Results directory: {RESULTS_DIR}")
    print(f"💾 Annotations file: {ANNOTATIONS_FILE}")
    print(f"🌐 Open your browser to: http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    main()
