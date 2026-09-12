# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image ingest helpers for the upload flow.

Supports single 2D modalities (X-ray, any DICOM single frame) as well as
multi-slice volumetric modalities (CT, MRI). Volumes are windowed and
down-sampled to a small number of slice images so they fit the MedGemma
prompt, following the same approach as the high-dimensional CT notebook.
DICOM requires pydicom; it is imported lazily so the rest of the app keeps
working even when only the plain-image path is used.
"""
import base64
import io
import logging
from pathlib import Path

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Modalities we can parse from the DICOM Modality tag (0x0008, 0x0060).
MODALITY_XRAY = {"CR", "DX", "XC", "RF", "MG"}
MODALITY_CT = {"CT"}
MODALITY_MRI = {"MR"}
MAX_PROMPT_IMAGES = 30  # Max slice images allowed in the prompt (UI clamps to this too).

# User-friendly modality label -> DICOM Modality tag values.
IDC_MODALITY_TAGS = {
    "X-ray": MODALITY_XRAY,
    "CT": MODALITY_CT,
    "MRI": MODALITY_MRI,
}

# Body parts we prefer per modality so the sample is a meaningful cancer
# study. Non-matching body parts are still accepted as a fallback.
IDC_PREFERRED_BODY_PART = {
    "X-ray": ("CHEST",),
    "CT": ("CHEST", "ABDOMEN", "PELVIS", "BRAIN"),
    "MRI": ("BRAIN", "BREAST", "PROSTATE"),
}

# Keep downloaded / sampled series small enough for the 4-bit 16 GB Colab GPU.
IDC_MAX_INSTANCES = 60
IDC_SIZE_MB_MIN = 1
IDC_SIZE_MB_MAX = 80

# CT collections contain scout / topogram / localizer series: 2D planar
# projections that look like plain X-rays but carry Modality=CT. Those would
# confuse the demo (a "CT" that looks like a chest X-ray), so exclude them.
IDC_EXCLUDED_DESCRIPTIONS = (
    "SCOUT", "TOPOGRAM", "TOPAGRAM", "LOCALIZER", "LOCALIZ",
    "SURVEY", "SCANOGRAM", "PLANAR", "TEXT"
)
# Minimum number of instances required per modality so a "CT" is really a
# volumetric series with enough slices to honor the chosen Slice count.
# MRI is exempt: many valid single-file multi-frame MR exams have 1 instance.
IDC_MIN_INSTANCES = {"X-ray": 1, "CT": 8, "MRI": 1}


def array_to_png_data_url(pixels):
    """Encode a 2D uint8 numpy array as a PNG data URL."""
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        arr = pixels
    else:
        arr = np.stack([pixels] * 3, axis=-1)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _stretch_to_uint8(arr, p_low=0.5, p_high=99.5):
    """Robust percentile-based stretch of any numeric array to uint8."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _apply_ct_window(hu, window_center, window_width):
    lower = float(window_center) - float(window_width) / 2.0
    scaled = (hu - lower) * (255.0 / float(window_width))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _slice_to_data_url(pixels, modality):
    """Convert one slice's raw pixels to a windowed PNG data URL."""
    if modality == "CT":
        # Default body window (level 40, width 400) unless DICOM provides one.
        pixels = _apply_ct_window(pixels, 40, 400)
    else:
        pixels = _stretch_to_uint8(pixels)
    return array_to_png_data_url(pixels)


def _to_hu(pixels, intercept=0, slope=1):
    return pixels.astype(np.float64) * slope + intercept


def _is_dicom_file(path):
    suffix = Path(path).suffix.lower()
    return suffix in {".dcm", ".dicom"}


def parse_dicom_slice(dicom_path, max_frames=MAX_PROMPT_IMAGES):
    """Read a DICOM file and return a list of windowed 2D slice dicts.

    One entry per *visible frame*: single-frame DICOM yields one entry, while
    multi-frame MRI/CT volumes (``pixel_array.ndim == 3``, e.g. Enhanced MR)
    are sampled down to ``max_frames`` evenly-spaced frames so they fit the
    MedGemma prompt and don't blow up memory.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        ds = pydicom.dcmread(str(dicom_path), force=False)
        pixel_array = ds.pixel_array
    except InvalidDicomError:
        raise ValueError(f"Not a readable DICOM file: {dicom_path}")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Error reading DICOM file {dicom_path}: {e}")

    modality = str(getattr(ds, "Modality", "")).strip().upper()
    if modality in MODALITY_CT:
        modality_name = "CT"
    elif modality in MODALITY_MRI:
        modality_name = "MRI"
    elif modality in MODALITY_XRAY:
        modality_name = "X-ray"
    else:
        modality_name = "X-ray" if pixel_array.ndim == 2 else "CT"

    # DICOM rescale (e.g. HU for CT) + window settings.
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    window_center, window_width = 40.0, 400.0
    if modality_name == "CT":
        center = getattr(ds, "WindowCenter", None)
        width = getattr(ds, "WindowWidth", None)
        try:
            center = float(np.atleast_1d(np.asarray(center, dtype=float))[0])
            width = float(np.atleast_1d(np.asarray(width, dtype=float))[0])
            if not (np.isfinite(center) and np.isfinite(width) and width > 0):
                raise ValueError
        except (TypeError, ValueError):
            center, width = 40.0, 400.0
        window_center, window_width = center, width

    arr = np.asarray(pixel_array)
    base_instance = int(getattr(ds, "InstanceNumber", 0) or 0)

    if arr.ndim == 3 and arr.shape[-1] != 3:
        frame_indices = np.unique(
            np.round(np.linspace(0, arr.shape[0] - 1,
                                 min(arr.shape[0], max_frames))).astype(int))
        frames = [arr[i] for i in frame_indices]
    else:
        frames = [arr]

    results = []
    for offset, frame in enumerate(frames):
        if modality_name == "CT":
            pixels = _apply_ct_window(_to_hu(frame, intercept, slope),
                                      window_center, window_width)
        else:
            pixels = _stretch_to_uint8(frame)
        results.append({"instance": base_instance + offset,
                        "modality": modality_name,
                        "pixels": pixels})
    return results


def collect_dicom_files(paths):
    """Expand uploaded files (zip + dcm) into a sorted DICOM path list."""
    import zipfile

    dicom_files = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(str(p.parent))
            for member in z.namelist():
                member_path = p.parent / member
                if member_path.is_file() and _is_dicom_file(member_path):
                    dicom_files.append(member_path)
        elif _is_dicom_file(p):
            dicom_files.append(p)
    # Stable sort by instance number when parseable, otherwise by path name.
    import re
    def _sort_key(path):
        instance = re.search(r"(?i)instance[_-]?(\d+)|(\d+)", path.stem)
        return (0, int(instance.group(1) or instance.group(2) or 0)) if instance \
            else (1, path.name)
    dicom_files.sort(key=_sort_key)
    return dicom_files


def sample_slices(parsed, max_slices=MAX_PROMPT_IMAGES):
    """Evenly sample a sorted slice list down to <= max_slices entries."""
    n = len(parsed)
    if n <= max_slices:
        return parsed
    idx = np.linspace(0, n - 1, max_slices).astype(int)
    return [parsed[i] for i in idx]


def process_plain_images(paths):
    """Build prompt-ready data URLs from plain 2D image files."""
    previews = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(
            buf.getvalue()).decode("ascii")
        previews.append({"label": Path(p).stem, "data_url": data_url})
    return previews


def process_upload_dicom(paths, max_slices=MAX_PROMPT_IMAGES):
    """Parse a DICOM series (or single frame) into windowed slice previews."""
    dicom_files = collect_dicom_files(paths)
    if not dicom_files:
        raise ValueError("No DICOM files (.dcm) found in the uploaded files.")
    if len(dicom_files) > 60:
        logger.warning("Large DICOM series (%d files); sampling for prompt.",
                       len(dicom_files))
    parsed = []
    for f in dicom_files:
        try:
            parsed.extend(parse_dicom_slice(f))
        except Exception as e:  # noqa: BLE001 - skip one bad file, keep the series
            logger.warning("Skipping unreadable DICOM file %s: %s", f, e)
    if not parsed:
        raise ValueError("No readable DICOM slices found in the uploaded files.")
    parsed.sort(key=lambda p: p["instance"])
    total = len(parsed)
    sliced = sample_slices(parsed, max_slices=max_slices)
    modalities = {p["modality"] for p in parsed}
    modality = "CT" if "CT" in modalities else (
               "MRI" if "MRI" in modalities else "X-ray")
    previews = []
    for i, p in enumerate(sliced, 1):
        data_url = array_to_png_data_url(p["pixels"])
        previews.append({"label": f"Slice {i}/{total}", "data_url": data_url})
    return {"modality": modality, "total_slices": total,
            "prompt_slices": len(previews), "previews": previews}


# --- NCI Imaging Data Commons (IDC) sample fetching ------------------------

def search_idc_series(modality, limit=3, preferred_body_parts=None):
    """Pick *limit* small public cancer series for *modality*.

    Uses the `idc-index` package (lazily imported) to query the ~100 TB of
    public cancer imaging data harmonized by the NCI Imaging Data Commons into
    DICOM. Returns metadata for series with few instances / small size so they
    download quickly and fit the MedGemma prompt.
    """
    if modality not in IDC_MODALITY_TAGS:
        raise ValueError(
            f"Unknown modality '{modality}'. Use X-ray, CT or MRI.")
    modality_tags = IDC_MODALITY_TAGS[modality]
    preferred = preferred_body_parts or IDC_PREFERRED_BODY_PART[modality]

    from idc_index import IDCClient
    client = IDCClient.client()

    in_mods = ", ".join(f"'{m}'" for m in modality_tags)
    in_body = ", ".join(f"'{b}'" for b in preferred)
    scout_filters = " AND ".join(
        f"UPPER(COALESCE(i.SeriesDescription, '')) NOT LIKE '%{s}%'"
        for s in IDC_EXCLUDED_DESCRIPTIONS)
    query = f"""
SELECT i.SeriesInstanceUID AS SeriesInstanceUID,
       i.collection_id AS collection_id,
       i.SeriesDescription AS SeriesDescription,
       i.BodyPartExamined AS BodyPartExamined,
       i.series_size_MB AS series_size_MB,
       (SELECT count(*) FROM index b
        WHERE b.SeriesInstanceUID = i.SeriesInstanceUID) AS n_instances
FROM index i
WHERE i.Modality IN ({in_mods})
  AND i.series_size_MB > {IDC_SIZE_MB_MIN}
  AND i.series_size_MB < {IDC_SIZE_MB_MAX}
  AND {scout_filters}
ORDER BY CASE WHEN i.BodyPartExamined IN ({in_body}) THEN 0 ELSE 1 END,
         random()
LIMIT 40
"""
    rows = client.sql_query(query)
    if rows is None or len(rows) == 0:
        raise ValueError(
            f"No public {modality} series found in IDC for this query.")
    min_instances = IDC_MIN_INSTANCES.get(modality, 1)
    infos = []
    for r in rows.to_dict("records"):
        instances = int(r.get("n_instances") or 0)
        if min_instances <= instances <= IDC_MAX_INSTANCES:
            infos.append({
                "series_uid": r["SeriesInstanceUID"],
                "collection_id": r.get("collection_id") or "",
                "series_description": r.get("SeriesDescription") or "",
                "body_part": r.get("BodyPartExamined") or "",
                "size_mb": float(r.get("series_size_MB") or 0),
                "instances": instances,
            })
        if len(infos) >= limit:
            break
    if not infos:
        raise ValueError(
            f"Only large {modality} series were found; none was small enough "
            f"(<= {IDC_MAX_INSTANCES} instances) for this demo.")
    return infos


def _download_series_from_gcs(client, series_uid, dest_dir):
    """Direct fallback: fetch the series file-by-file over the public GCS bucket."""
    rows = client.sql_query(
        f"SELECT gcs_url FROM index WHERE SeriesInstanceUID = '{series_uid}'")
    if rows is None or len(rows) == 0:
        raise ValueError(
            f"No IDC file URLs found for series {series_uid}.")
    for i, r in enumerate(rows.to_dict("records")):
        url = r.get("gcs_url")
        if not url:
            continue
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out_path = dest_dir / f"{i:04d}.dcm"
        out_path.write_bytes(resp.content)
    logger.info("Downloaded %d files for IDC series %s via GCS.",
                i + 1, series_uid)


def _find_dicom_paths(root):
    """Recursively find DICOM files (handles zip archives and extension-less
    files by peeking at the DICOM header)."""
    import zipfile

    import pydicom

    root = Path(root)
    for zf in list(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zf) as z:
                z.extractall(root)
        except zipfile.BadZipFile:  # noqa: PERF203
            continue

    candidates = list(root.rglob("*.dcm"))
    candidates += [f for f in root.rglob("*")
                   if f.is_file() and f.suffix.lower()
                   not in {".zip", ".png", ".jpg", ".jpeg", ".webp", ".gif"}]
    found = []
    for f in candidates:
        try:
            pydicom.dcmread(str(f), force=True, stop_before_pixels=True)
            found.append(f)
        except Exception:  # noqa: BLE001 - not a DICOM file
            continue
    return found


def download_idc_series(series_uid, dest_dir):
    """Download a series' DICOM files into dest_dir and return their paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    from idc_index import IDCClient
    client = IDCClient.client()
    try:
        client.download_dicom_series(
            seriesInstanceUID=[series_uid], downloadDir=str(dest_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning("idc-index download failed (%s); "
                       "falling back to direct GCS fetch.", e)
        _download_series_from_gcs(client, series_uid, dest_dir)

    files = _find_dicom_paths(dest_dir)
    if not files:
        raise ValueError(
            f"No readable DICOM files were downloaded for series {series_uid}.")
    return files


def fetch_idc_sample(modality, dest_dir=None):
    """Download one small public cancer sample series of the requested
    modality and return (series_metadata, list_of_dicom_paths)."""
    import tempfile

    info = search_idc_series(modality, limit=1)[0]
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(
            prefix=f"radexplain-idc-{info['series_uid'][:8]}-"))
    files = download_idc_series(info["series_uid"], dest_dir)
    return info, files


def fetch_idc_sample_pool(modality, count, dest_root, max_preview_slices=6):
    """Download and parse *count* small series of *modality* into a pool.

    Persists each series under ``dest_root/<series_uid>`` and returns a list
    of sample dicts holding the metadata, the local file paths and up to
    ``max_preview_slices`` display previews. Undecodable series are skipped
    (with a warning) instead of failing the whole batch.
    """
    import tempfile

    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    infos = search_idc_series(modality, limit=count)
    samples = []
    for info in infos:
        dir_name = info["series_uid"].replace(".", "")[:28]
        series_dir = dest_root / dir_name
        try:
            files = download_idc_series(info["series_uid"], series_dir)
            parsed = process_upload_dicom(files, max_slices=max_preview_slices)
            samples.append({
                "modality": modality,
                "info": info,
                "files": files,
                "dir": str(series_dir),
                "total_slices": parsed["total_slices"],
                "previews": parsed["previews"],
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping IDC %s series %s: %s",
                           modality, info["series_uid"], e)
            continue
    return samples