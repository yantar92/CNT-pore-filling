# MIT License
# Copyright (c) 2026 Ihor Radchenko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Plotting infrastructure: style setup, data loading, and disk caching.

Does NOT contain actual plot functions -- those live in individual
analysis scripts.  This module provides shared building blocks:

- ``setup_mpl_style``: consistent matplotlib rcParams for publication figures.
- ``load_results_df``: read scan results CSV with caching.
- ``load_timeseries_df``: read and combine time-series CSV files with caching.
"""

import gzip
import hashlib
import logging
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import pandas as pd

logger = logging.getLogger(__name__)

# Column names used when reading time-series CSV files produced by
# save_timeseries_csv (the header row is skipped, so these are
# assigned on read).
TIMESERIES_COLUMNS: list[str] = ['Time', 'Filling', 'Formation energy']


# --- Matplotlib style ----------------------------------------------------

def setup_mpl_style(
        *,
        font_size: float = 12,
        width: float = 4.13,
        ratio: float = 0.75,
        dpi: int = 300,
        savefig_dpi: int = 600,
) -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Applies common settings: editable PDF text, direction-in ticks,
    minor ticks, and consistent font/line sizing.  Scripts that need
    further customization (e.g. a seaborn base style) should call this
    function *after* any style sheet import so these values take
    precedence.

    Args:
        font_size: Base font size in pt.
        width: Figure width in inches (default 4.13 = half A4).
        ratio: Height / width ratio (default 0.75 = 3:4).
        dpi: Screen DPI for interactive display.
        savefig_dpi: DPI for saved figures.
    """
    height = width * ratio
    mpl.rcParams.update({
        'figure.figsize': (width, height),
        'figure.dpi': dpi,
        'savefig.dpi': savefig_dpi,

        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size * 4 / 3,
        'legend.fontsize': font_size / 1.2,
        'xtick.labelsize': font_size / 1.2,
        'ytick.labelsize': font_size / 1.2,

        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        'lines.linewidth': 1.2,
        'lines.markersize': 3.5,

        'pdf.fonttype': 42,          # editable text in Illustrator
        'ps.fonttype': 42,
        'mathtext.default': 'regular',
    })


# --- Data loading --------------------------------------------------------

def load_results_df(path: str | Path) -> pd.DataFrame:
    """Load scan results CSV with standard column names.

    The CSV must have no header row; columns are assigned from
    ``RESULTS_CSV_COLUMNS`` (defined in ``mcpore.core``).  Results are
    cached to disk beside the CSV file (``<path>.pkl.gz``) using
    content-based invalidation.

    Args:
        path: Path to the results CSV file.

    Returns:
        DataFrame with the standard scan-result schema.
        The DataFrame carries a ``_from_cache`` attribute (bool)
        indicating whether the data was read from a disk cache.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from mcpore.core import RESULTS_CSV_COLUMNS

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Results file not found: {path}')

    cache_path = path.with_suffix('.pkl.gz')

    def _compute() -> pd.DataFrame:
        df = pd.read_csv(path, names=RESULTS_CSV_COLUMNS)
        logger.info('Loaded %d rows from %s', len(df), path)
        return df

    df, from_cache = _cached_dataframe(_compute, cache_path, source_paths=[path])
    df._from_cache = from_cache
    return df


def load_timeseries_df(
        data_dir: str | Path,
        *,
        file_glob: str,
        n_samples: int | None = None,
        downsample: int = 1,
        concatenate: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Load time-series CSV files matching a glob pattern.

    Each file is read with ``TIMESERIES_COLUMNS`` (skipping the first
    row).

    Results are cached to disk inside *data_dir* using content-based
    invalidation (keyed on the matched files' mtimes).

    Args:
        data_dir: Directory containing the per-replicate CSV files.
        file_glob: Glob pattern to match files (e.g.
            ``'0.00V_2200K_15A_r*.csv.gz'``).  The pattern is
            appended to *data_dir*.
        n_samples: If set, read at most this many files.
        downsample: Keep every N-th row (applied per file before any
            concatenation).
        concatenate: If ``True`` (default), concatenate all replicates
            into a single DataFrame sorted by ``Time``.  If ``False``,
            return a list of DataFrames, one per file, in the order
            they were matched.

    Returns:
        Concatenated, sorted DataFrame when ``concatenate=True``;
        list of per-file DataFrames when ``concatenate=False``.
        Each DataFrame carries a ``_from_cache`` attribute (bool)
        indicating whether the data was read from a disk cache.

    Raises:
        FileNotFoundError: If no files match the glob pattern.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(file_glob))

    if not files:
        raise FileNotFoundError(
            f'No time-series files matching: {data_dir / file_glob}')

    if n_samples is not None:
        files = files[:n_samples]

    # Derive a cache name from the glob and parameters.
    # When concatenate=False, use a distinct suffix so the two modes
    # do not collide; the default (concatenate=True) keeps the
    # existing cache name for backward compatibility.
    glob_slug = file_glob.replace('*', 'STAR').replace('/', '_')
    cache_name = f'{glob_slug}_n{len(files)}_d{downsample}'
    if not concatenate:
        cache_name += '_list'
    cache_path = data_dir / (cache_name + '.pkl.gz')

    def _compute() -> pd.DataFrame | list[pd.DataFrame]:
        dfs: list[pd.DataFrame] = []
        for f in files:
            logger.debug('Reading %s', f)
            try:
                df0 = pd.read_csv(
                    f, names=TIMESERIES_COLUMNS, skiprows=1)
                df0 = df0.iloc[::downsample]
            except (pd.errors.ParserError, OSError, ValueError, EOFError) as e:
                logger.warning('Skipping %s: %s', f, e)
                continue
            dfs.append(df0)
        if not dfs:
            raise RuntimeError('No data read from matched files')
        if concatenate:
            result = pd.concat(dfs, ignore_index=True)
            result = result.sort_values(
                by=['Time']).reset_index(drop=True)
            logger.info(
                'Loaded %d rows from %d files (downsample=%d)',
                len(result), len(files), downsample)
            return result
        logger.info(
            'Loaded %d files (downsample=%d)',
            len(dfs), downsample)
        return dfs

    result, from_cache = _cached_dataframe(_compute, cache_path, source_paths=files)
    if concatenate:
        result._from_cache = from_cache
    else:
        for df in result:
            df._from_cache = from_cache
    return result


# --- Caching internals ---------------------------------------------------

def _compute_source_hash(paths: list[Path]) -> str:
    """Return an MD5 hex digest of the mtimes of all *paths*.

    If a path is a directory, its immediate children are included
    recursively (one level).  This is a simple invalidation heuristic:
    any file touched in the source tree invalidates the cache.
    """
    hasher = hashlib.md5()
    for p in sorted(paths):
        if p.is_dir():
            for child in sorted(p.iterdir()):
                stat = child.stat()
                hasher.update(
                    f'{child}:{stat.st_mtime}:{stat.st_size}'.encode())
        elif p.exists():
            stat = p.stat()
            hasher.update(f'{p}:{stat.st_mtime}:{stat.st_size}'.encode())
        else:
            # Missing path: produce a unique hash so the cache is
            # invalidated and the caller handles the error.
            hasher.update(f'{p}:MISSING:{os.urandom(8)}'.encode())
    return hasher.hexdigest()


def _cached_dataframe(
        compute_fn: Callable[[], pd.DataFrame],
        cache_path: Path,
        *,
        source_paths: list[Path],
) -> tuple[pd.DataFrame, bool]:
    """Compute or retrieve a DataFrame from a disk cache.

    Uses content-based invalidation: the cache is reused only when
    the mtime/size hash of *source_paths* matches the stored hash.
    Writes are atomic (temp file + rename).

    Args:
        compute_fn: Callable that produces the DataFrame from scratch.
        cache_path: Path to the cache file (``.pkl.gz``).
        source_paths: Files/directories whose modification signals
            determine cache validity.

    Returns:
        Tuple of (DataFrame, from_cache: bool).
    """
    source_hash = _compute_source_hash(source_paths)

    if cache_path.exists():
        try:
            with gzip.open(cache_path, 'rb') as f:
                cached_hash, cached_data = pickle.load(f)
            if cached_hash == source_hash:
                logger.info('Cache hit: %s', cache_path)
                return cached_data, True
            logger.info('Cache stale: %s', cache_path)
        except (OSError, pickle.PickleError, EOFError) as e:
            logger.warning('Failed to read cache %s: %s', cache_path, e)

    data = compute_fn()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode='wb', dir=cache_path.parent, delete=False,
            suffix='.tmp.gz') as tmp:
        try:
            with gzip.open(tmp.name, 'wb') as f:
                pickle.dump(
                    (source_hash, data), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            os.unlink(tmp.name)
            raise
    os.rename(tmp.name, cache_path)
    logger.debug('Cached to %s', cache_path)
    return data, False
