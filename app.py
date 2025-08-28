# TaleTrain Viewer — Streamlit app
# Requirements: `pip install streamlit pandas`
# Optional: `ffmpeg` binary on PATH for fast clip extraction (recommended)

import os
import re
import csv
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Utilities
# -----------------------------

REMAP_ID = {
    'P01': 'P01', 'P02': 'P02', 'P03': 'P03', 'P04': 'P04',
    'P06': 'P05', 'P07': 'P06', 'P08': 'P07', 'P09': 'P08', 'P10': 'P09',
    'P11': 'P10', 'P13': 'P11', 'P14': 'P12', 'P15': 'P13', 'P16': 'P14',
    'P17': 'P15', 'P18': 'P16', 'P19': 'P17', 'P21': 'P18', 'P22': 'P19', 'P23': 'P20',
}

THEME_ORDER = [
    'engagement-pros', 'engagement-cons', 'engagement-suggest',
    'narrative-pros', 'narrative-cons', 'narrative-suggest'
]

TIME_HHMMSS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[\.]?(\d{0,3})$")

def hhmmss_to_seconds(s: str) -> float:
    """Convert 'HH:MM:SS.mmm' -> seconds (float). Safe for missing millis."""
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return math.nan
    s = s.strip()
    m = TIME_HHMMSS_RE.match(s)
    if not m:
        # Try pandas to_timedelta as a fallback
        try:
            return pd.to_timedelta(s).total_seconds()
        except Exception:
            return math.nan
    hh, mm, ss, ms = m.groups()
    ms = (ms + '000')[:3] if ms else '000'
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def detect_theme_columns(df: pd.DataFrame) -> list:
    cols = df.columns.tolist()
    # everything strictly between 'default' and 'context'
    start_idx = cols.index('default') + 1
    end_idx = cols.index('context')
    theme_cols = cols[start_idx:end_idx]
    return theme_cols


def read_tsv_loose(path: str) -> pd.DataFrame:
    """Robust TSV reader that treats quotes literally and skips malformed lines if needed."""
    try:
        return pd.read_csv(
            path, sep='\t', comment='#', engine='python',
            quoting=csv.QUOTE_NONE, escapechar='\\', na_filter=False
        )
    except pd.errors.ParserError:
        return pd.read_csv(
            path, sep='\t', comment='#', engine='python',
            quoting=csv.QUOTE_NONE, escapechar='\\', na_filter=False,
            on_bad_lines='skip'
        )


def get_data_summary(file_path: str, video_ext: str = ".mp4") -> pd.DataFrame:
    # Infer experiment name from filename (e.g., P11_TaleTrain.txt -> P11_TaleTrain)
    experiment_name = os.path.basename(file_path).rsplit('.', 1)[0]

    df = read_tsv_loose(file_path)

    # Identify theme columns dynamically
    theme_cols = detect_theme_columns(df)

    # Normalize and stack themes -> long form
    themes = (
        df[theme_cols]
        .astype(str)
        .replace('\u00A0', ' ', regex=False)
        .apply(lambda s: s.str.strip())
        .replace({'': pd.NA})
    )

    long = (
        themes
        .stack(dropna=True)
        .rename_axis(index=['row_idx', 'theme'])
        .reset_index(name='contents')
    )

    # Bring meta columns (start/end & context). Prefer hh:mm:ss.ms for display
    start_col = '시작 시간 - hh:mm:ss.ms'
    end_col   = '종료 시간 - hh:mm:ss.ms'
    if start_col not in df.columns or end_col not in df.columns:
        # fallback to ss.msec if needed
        start_col = '시작 시간 - ss.msec'
        end_col   = '종료 시간 - ss.msec'

    meta = df[[c for c in [start_col, end_col, 'context'] if c in df.columns]].copy()
    meta = meta.rename(columns={start_col: 'start_time', end_col: 'end_time'})

    merged = long.merge(meta, left_on='row_idx', right_index=True, how='left')

    # Participant/condition from filename like P11_TaleTrain
    if '_' in experiment_name:
        participant, condition = experiment_name.split('_', 1)
    else:
        # if no underscore, try dash or just the whole name
        parts = re.split(r'[- ]', experiment_name, maxsplit=1)
        participant = parts[0]
        condition = parts[1] if len(parts) > 1 else 'Unknown'

    participant_mapped = REMAP_ID.get(participant, participant)

    merged['participant'] = participant_mapped
    merged['condition'] = condition

    # Sorting by start/end (convert strings to seconds if in hh:mm:ss.ms)
    merged['start_sec'] = merged['start_time'].apply(hhmmss_to_seconds)
    merged['end_sec'] = merged['end_time'].apply(hhmmss_to_seconds)

    # Categorical sort for theme (optional, keeps a coherent order)
    # present = [t for t in THEME_ORDER if t in merged['theme'].unique()]
    # merged['theme'] = pd.Categorical(merged['theme'], categories=present, ordered=True)

    merged = merged.sort_values(['start_sec', 'end_sec', 'theme'], kind='mergesort').reset_index(drop=True)

    # Final column order
    final_df = merged[['start_time', 'end_time', 'start_sec', 'end_sec', 'theme', 'contents', 'context', 'participant', 'condition']].copy()
    final_df.insert(0, 'index', range(len(final_df)))
    return final_df


def get_all_data_summary(data_dir: str) -> pd.DataFrame:
    dfs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith('.txt'):
            path = os.path.join(data_dir, fname)
            try:
                dfs.append(get_data_summary(path))
            except Exception as e:
                st.warning(f"Failed to parse {fname}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# Video helpers
# -----------------------------

def seconds_to_hhmmss(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or math.isnan(seconds):
        return "00:00:00.000"
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_clip(input_path: Path, start: float, end: float, out_dir: Path) -> Path | None:
    """Create (or reuse) a clipped segment via ffmpeg -c copy. Returns output path or None if ffmpeg not available."""
    if shutil.which('ffmpeg') is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = input_path.stem
    out_path = out_dir / f"{safe_name}_{int(start*1000)}_{int(end*1000)}.mp4"
    if out_path.exists():
        return out_path
    # Use -ss before -i for fast seek, -to for end timestamp (absolute from start of input)
    start_str = seconds_to_hhmmss(start)
    end_str = seconds_to_hhmmss(end)
    cmd = [
        'ffmpeg', '-y', '-ss', start_str, '-to', end_str,
        '-i', str(input_path), '-c', 'copy', str(out_path)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return out_path
    except subprocess.CalledProcessError:
        return None


def embed_seek_player(video_path: Path, start: float, end: float, height: int = 360):
    """Embed an HTML5 player that seeks to [start, end] without clipping (fallback when ffmpeg not present)."""
    # Streamlit can serve local files via st.video, but for JS control we embed a base64 path using st.file_uploader-like hack is heavy.
    # Instead, rely on st.video for simple playback + instructions.
    st.info("ffmpeg not found — showing full video. Use the seek buttons to jump.")
    st.video(str(video_path))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏩ Seek to start", use_container_width=True):
            st.session_state["seek_to"] = start
    with c2:
        st.write(f"End at {seconds_to_hhmmss(end)} (manual pause)")


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="TaleTrain Viewer", layout="wide")

st.title("🎬 TaleTrain Viewer")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data folder (TSVs)", value="./label")
    video_dir = st.text_input("Video folder (MP4s)", value="./video")
    show_table = st.checkbox("Show raw table", value=True)
    st.caption("Expect files like P11_TaleTrain.txt and videos like P11_TaleTrain.mp4")

# Load data
if not os.path.isdir(data_dir):
    st.error(f"Data folder not found: {data_dir}")
    st.stop()

df = get_all_data_summary(data_dir)
if df.empty:
    st.warning("No .txt files found or parsed.")
    st.stop()

# Left: selector panel
left, right = st.columns([0.48, 0.52])

with left:
    st.subheader("Entries")

    # Compact summary for listing
    list_df = df[['index', 'participant', 'condition', 'start_time', 'end_time', 'theme', 'contents']].copy()
    list_df['label'] = list_df.apply(lambda r: f"{r['participant']}{r['condition']} | {r['theme']} / {r['contents']}", axis=1)

    # Selection widget
    default_idx = int(st.session_state.get('selected_row', 0))
    selected_label = st.selectbox("Select a row", options=list_df['label'].tolist(), index=min(default_idx, len(list_df)-1))
    selected_row = list_df[list_df['label'] == selected_label].iloc[0]
    st.session_state['selected_row'] = int(selected_row['index'])

    # Optional raw table
    if show_table:
        st.dataframe(df)

with right:
    st.subheader("Player")
    row = df[df['index'] == selected_row['index']].iloc[0]

    # Show details
    st.write(f"{row['participant']} - {row['condition']}")
    st.markdown(f"**{row['theme']}**: {row['contents']}")
    st.markdown(f"**Context**: {row.get('context', '')}")

    # Video path
    video_name = f"{'P11'}_{row['condition']}.mp4"
    video_path = Path(video_dir) / video_name

    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
    else:
        start_s = float(row['start_sec']) if pd.notna(row['start_sec']) else 0.0
        end_s = float(row['end_sec']) if pd.notna(row['end_sec']) else max(0.1, start_s + 5.0)
        if end_s <= start_s:
            end_s = start_s + 0.1

        clip_dir = Path('.clips')
        clip_path = ensure_clip(video_path, start_s, end_s, clip_dir)
        if clip_path is not None and clip_path.exists():
            st.video(str(clip_path))
            st.caption(f"Clip: {seconds_to_hhmmss(start_s)} → {seconds_to_hhmmss(end_s)} (ffmpeg)")
        else:
            embed_seek_player(video_path, start_s, end_s)

st.caption("Tip: Install ffmpeg for fast, frame-accurate clipping. On macOS: `brew install ffmpeg`")

# Run locally with: streamlit run app.py
