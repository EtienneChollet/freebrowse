"""Build a self-contained HTML comparing synth-only vs synth-real predictions.

For each sample folder in debug_output/, pick `prompt_iter_000` and embed both
prediction overlay PNGs (base64) side by side: synth-only on the left,
synth-real on the right. One row per sample.
"""
import base64
import sys
from pathlib import Path
from typing import Optional


DEBUG_ROOT = Path('/autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim/debug_output')
OUTPUT_HTML = Path('/autofs/cluster/dalcalab2/users/epc28/projects/freebrowse_bhim/prediction_comparison.html')

PROMPT_ITER_NAME = 'prompt_iter_000'
LEFT_NAME = 'prediction_overlay_neel_synth-only.png'
RIGHT_NAME = 'prediction_overlay_neel_synth-real.png'


def encode_png(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:image/png;base64,{data}'


def first_prompt_iter(session_dir: Path) -> Optional[Path]:
    iter_dir = session_dir / PROMPT_ITER_NAME
    return iter_dir if iter_dir.is_dir() else None


def collect_rows():
    rows = []
    for session_dir in sorted(DEBUG_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue
        iter_dir = first_prompt_iter(session_dir)
        if iter_dir is None:
            continue
        left = iter_dir / LEFT_NAME
        right = iter_dir / RIGHT_NAME
        if not (left.exists() and right.exists()):
            print(f'SKIP (missing PNG): {session_dir.name}', file=sys.stderr)
            continue
        rows.append({
            'session': session_dir.name,
            'iter': iter_dir.name,
            'left_src': encode_png(left),
            'right_src': encode_png(right),
        })
    return rows


def render_html(rows) -> str:
    row_html = []
    for r in rows:
        row_html.append(f'''
    <div class="row">
      <div class="header">
        <span class="session">{r['session']}</span>
        <span class="iter">{r['iter']}</span>
      </div>
      <div class="pair">
        <figure>
          <figcaption>neel_synth-only</figcaption>
          <img src="{r['left_src']}" alt="synth-only: {r['session']}" />
        </figure>
        <figure>
          <figcaption>neel_synth-real</figcaption>
          <img src="{r['right_src']}" alt="synth-real: {r['session']}" />
        </figure>
      </div>
    </div>''')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Prediction comparison: synth-only vs synth-real</title>
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
    font-family: ui-monospace, Menlo, monospace;
    font-size: 12px;
    color: #aaa;
    margin-bottom: 8px;
  }}
  .session {{ color: #ddd; }}
  .pair {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  figure {{ margin: 0; }}
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
  <h1>Prediction comparison: synth-only (left) vs synth-real (right)</h1>
  <div class="sub">{len(rows)} samples &middot; first prompt iter per session</div>
{''.join(row_html)}
</body>
</html>
'''


def main():
    rows = collect_rows()
    print(f'Collected {len(rows)} rows', file=sys.stderr)
    OUTPUT_HTML.write_text(render_html(rows))
    size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
    print(f'Wrote {OUTPUT_HTML} ({size_mb:.1f} MB)', file=sys.stderr)


if __name__ == '__main__':
    main()
