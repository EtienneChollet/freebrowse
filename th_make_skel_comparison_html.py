"""Build a standalone HTML comparing EPC-2141 skeleton model predictions.

For each saved prompt iteration in `debug_output/`, embed the three standard
prediction overlay PNGs side by side:

- skel-dilate
- skel-only
- skel-baseline
"""

import base64
import html
import sys
from pathlib import Path
from typing import Dict, List

DEBUG_ROOT = Path("/autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim/debug_output")
OUTPUT_HTML = Path(
    "/autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim/"
    "prediction_comparison_skel.html"
)
MODEL_NAMES = ["skel-dilate", "skel-only", "skel-baseline"]


def encode_png(path: Path) -> str:
    """Return a PNG file as a base64 data URI."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def prompt_iteration_dirs() -> List[Path]:
    """Return all saved FreeBrowse debug prompt-iteration directories."""
    if not DEBUG_ROOT.exists():
        return []
    return sorted(path for path in DEBUG_ROOT.rglob("prompt_iter_*") if path.is_dir())


def collect_rows() -> List[Dict[str, str]]:
    """Collect rows where all three skeleton model overlays exist."""
    rows: List[Dict[str, str]] = []
    for prompt_dir in prompt_iteration_dirs():
        image_paths = {
            model_name: prompt_dir / f"prediction_overlay_{model_name}.png"
            for model_name in MODEL_NAMES
        }
        missing = [model_name for model_name, path in image_paths.items() if not path.exists()]
        if missing:
            label = prompt_dir.relative_to(DEBUG_ROOT)
            print(f"SKIP {label}: missing {', '.join(missing)}", file=sys.stderr)
            continue

        row: Dict[str, str] = {
            "label": str(prompt_dir.relative_to(DEBUG_ROOT)),
            "session": prompt_dir.parent.name,
            "iter": prompt_dir.name,
        }
        for model_name, path in image_paths.items():
            row[model_name] = encode_png(path)
        rows.append(row)
    return rows


def render_row(row: Dict[str, str]) -> str:
    """Render one comparison row."""
    figures = []
    for model_name in MODEL_NAMES:
        escaped_model = html.escape(model_name)
        escaped_label = html.escape(row["label"])
        figures.append(
            f"""
        <figure>
          <figcaption>{escaped_model}</figcaption>
          <img loading="lazy" src="{row[model_name]}" alt="{escaped_model}: {escaped_label}" />
        </figure>"""
        )

    return f"""
    <div class="row">
      <div class="header">
        <span class="session">{html.escape(row['session'])}</span>
        <span class="iter">{html.escape(row['iter'])}</span>
      </div>
      <div class="model-stack">{''.join(figures)}
      </div>
    </div>"""


def render_html(rows: List[Dict[str, str]]) -> str:
    """Render the full standalone HTML document."""
    row_html = "".join(render_row(row) for row in rows)
    model_header = " vs ".join(MODEL_NAMES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>EPC-2141 prediction comparison: {html.escape(model_header)}</title>
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #111;
    color: #eee;
    font-family: -apple-system, system-ui, sans-serif;
  }}
  h1 {{ font-size: 20px; margin: 0 0 16px; }}
  .sub {{ color: #999; font-size: 13px; margin-bottom: 24px; }}
  .row {{
    margin-bottom: 32px;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 12px;
    background: #1a1a1a;
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12px;
    color: #aaa;
    margin-bottom: 8px;
  }}
  .session {{ color: #ddd; overflow-wrap: anywhere; }}
  .iter {{ white-space: nowrap; }}
  .model-stack {{
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
  }}
  figure {{
    margin: 0;
    border-top: 1px solid #333;
    padding-top: 12px;
  }}
  figure:first-child {{
    border-top: 0;
    padding-top: 0;
  }}
  figcaption {{
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12px;
    color: #9cf;
    text-align: center;
    margin-bottom: 4px;
  }}
  img {{
    width: 100%;
    height: auto;
    display: block;
    background: #000;
    border-radius: 4px;
  }}
</style>
</head>
<body>
  <h1>EPC-2141 prediction comparison: {html.escape(model_header)}</h1>
  <div class="sub">{len(rows)} prompt iterations &middot; standard overlays only</div>
{row_html}
</body>
</html>
"""


def main() -> None:
    """Build the HTML report."""
    rows = collect_rows()
    print(f"Collected {len(rows)} rows", file=sys.stderr)
    OUTPUT_HTML.write_text(render_html(rows))
    size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT_HTML} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
