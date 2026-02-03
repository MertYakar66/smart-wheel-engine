"""
Metadata fingerprinting for reproducibility and audit trails.
"""
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_run_metadata(
    config: Optional[Dict[str, Any]] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate metadata for reproducibility.

    Args:
        config: Configuration parameters used for this run
        data_start: Start date of data range
        data_end: End date of data range

    Returns:
        Metadata dictionary
    """
    return {
        '_metadata_version': '1.0',
        '_git_hash': get_git_hash(),
        '_git_branch': get_git_branch(),
        '_generated_at': datetime.utcnow().isoformat() + 'Z',
        '_data_start': data_start,
        '_data_end': data_end,
        '_config': config or {},
    }


def embed_metadata_in_df(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Embed metadata in DataFrame attrs (preserved in parquet format).

    Args:
        df: DataFrame to annotate
        metadata: Metadata dictionary

    Returns:
        DataFrame with metadata attached
    """
    df = df.copy()
    df.attrs.update(metadata)
    return df


def save_with_metadata(
    df: pd.DataFrame,
    filepath: str,
    config: Optional[Dict[str, Any]] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
    format: str = 'csv'
) -> None:
    """
    Save DataFrame with metadata sidecar file.

    For CSV: creates a companion .meta.json file
    For Parquet: embeds metadata in file attrs

    Args:
        df: DataFrame to save
        filepath: Output file path
        config: Configuration parameters
        data_start: Data range start
        data_end: Data range end
        format: 'csv' or 'parquet'
    """
    metadata = get_run_metadata(config, data_start, data_end)
    metadata['_row_count'] = len(df)
    metadata['_columns'] = list(df.columns)

    filepath = Path(filepath)

    if format == 'parquet':
        df = embed_metadata_in_df(df, metadata)
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
        # Write companion metadata file
        meta_path = filepath.with_suffix(filepath.suffix + '.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from sidecar file or parquet attrs.

    Args:
        filepath: Data file path

    Returns:
        Metadata dictionary or None
    """
    filepath = Path(filepath)

    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
        return dict(df.attrs) if df.attrs else None
    else:
        meta_path = filepath.with_suffix(filepath.suffix + '.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
    return None
