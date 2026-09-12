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
import os
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


def _max_slice_image_side():
    val = os.environ.get("MEDGEMMA_IMAGE_SIDE", "384")
    try:
        val = int(val)
    except ValueError:
        return 384
    return val if val > 0 else None

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
# Preference band for the demo; the query orders by body-part fit then random
# and only series within [min, max] instances qualify via the SQL filter.
IDC_MAX_INSTANCES = 400
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
    """Encode a 2D uint8 numpy array as a PNG data URL.

    Slice images are down-scaled to at most ``MAX_SLICE_IMAGE_SIDE`` pixels on
    the longest side before encoding. That caps the number of SigLIP image
    tokens per slice (~4x fewer at 384 px vs 512 px), which is what keeps the
    image prefill from out-of-memorying a 16 GB T4 when several slices are
    sent at once. Override with MEDGEMMA_IMAGE_SIDE.
    """
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        arr = pixels
    else:
        arr = np.stack([pixels] * 3, axis=-1)
    img = Image.fromarray(arr)
    side = _max_slice_image_side()
    if side is not None and max(img.size) > side:
        img.thumbnail((side, side), Image.LANCZOS)
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


def process_plain_images(paths, max_slices=MAX_PROMPT_IMAGES):
    """Build prompt-ready data URLs from plain 2D image files as a series.

    Each uploaded image is one "slice" of the study. Returns the same result
    shape as ``process_upload_dicom`` (modality/total_slices/prompt_slices/
    previews) so both branches behave the same in the API.
    """
    previews = []
    skipped = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()).decode("ascii")
            previews.append({"label": Path(p).stem, "data_url": data_url})
        except Exception as e:  # noqa: BLE001 - skip one bad file
            skipped.append(f"{Path(p).name} ({e})")
    if not previews:
        raise ValueError("No readable image files were found in the upload.")
    total = len(previews)
    if total > max_slices:
        idx = np.linspace(0, total - 1, max_slices).astype(int)
        previews = [previews[i] for i in idx]
    return {"modality": "X-ray", "total_slices": total,
            "prompt_slices": len(previews), "previews": previews,
            "skipped_files": skipped}


def _read_dicom_frames(path):
    """Read one DICOM file and return raw (un-windowed) 2D frame arrays.

    Single-frame files yield one entry; multi-frame files (e.g. Enhanced MR)
    yield one entry per frame. Raw pixel values are kept so CT can be
    straightened to real HU afterwards and then *de-calibrated* by the fixed
    windowing, instead of trusting the hitting DICOM WindowCenter/Width.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        ds = pydicom.dcmread(str(path), force=False)
        pixel_array = ds.pixel_array
    except InvalidDicomError:
        raise ValueError(f"Not a readable DICOM file: {path}")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Error reading DICOM file {path}: {e}")

    arr = np.asarray(pixel_array)
    modality = str(getattr(ds, "Modality", "")).strip().upper()
    if arr.ndim == 3 and arr.shape[-1] != 3:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        frames = [arr]

    base_inst = int(getattr(ds, "InstanceNumber", 0) or 0)
    slice_loc = getattr(ds, "SliceLocation", None)
    try:
        slice_loc = float(np.atleast_1d(np.asarray(slice_loc, dtype=float))[0])
    except (TypeError, ValueError):
        slice_loc = float("nan")
    spacing = getattr(ds, "PixelSpacing", None)
    try:
        spacing = [float(x) for x in np.atleast_1d(
            np.asarray(spacing, dtype=float))]
    except (TypeError, ValueError):
        spacing = None

    common = {
        "modality": modality,
        "series_desc": str(getattr(ds, "SeriesDescription", "") or ""),
        "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
        "slice_loc": slice_loc,
        "spacing": spacing,
        "slope": float(getattr(ds, "RescaleSlope", 1) or 1),
        "intercept": float(getattr(ds, "RescaleIntercept", 0) or 0),
    }
    return [{"array": f, "instance": base_inst + fi, "frame": fi, **common}
            for fi, f in enumerate(frames)]


def _sort_frames(frames):
    def _key(f):
        loc = f["slice_loc"]
        if np.isfinite(loc):
            return (0, loc, f["instance"], f["frame"])
        return (1, f["instance"], f["frame"])
    return sorted(frames, key=_key)


def _mask_empty_slices(volume):
    """Drop near-constant (empty / air-only / padding) slices from a volume."""
    n = volume.shape[0]
    if n <= 2:
        return np.arange(n)
    keep = []
    for z in range(n):
        s = volume[z]
        p0 = np.percentile(s, 0.5)
        p99 = np.percentile(s, 99.5)
        peak = float(((s >= p99) | (s <= p0)).mean())
        if peak <= 0.995:
            keep.append(z)
    if not keep:
        keep = list(range(n))
    return np.array(keep, dtype=int)


def _decalibrate_volume(volume, is_ct):
    """Return intensity-normalized float volume (removes calibration).

    For CT the raw HU are clipped to a fixed anatomical window and normalized
    to [0, 1]; everything else is percentile-stretched. Attention and empty-
    slice logic operate on structure, not on vendor calibration.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if is_ct:
        lo, hi = -175.0, 375.0
        volume = np.clip(volume, lo, hi)
    lo, hi = np.percentile(volume, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(volume)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0)


def _slice_edge_energy(s):
    gy, gx = np.gradient(s)
    return float(np.hypot(gx, gy).mean())


def _slice_entropy(s):
    hist, _ = np.histogram(s, bins=64, range=(0.0, 1.0))
    p = hist / hist.sum()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum() / np.log2(64))


def _slice_non_background(s):
    return float((s > 0.05).mean())


def _slice_similarity(a, b):
    """Normalized cross-correlation between two slices at 16x16 resolution."""
    from PIL import Image
    ta = np.asarray(Image.fromarray(a).resize((16, 16), Image.LANCZOS),
                    dtype=np.float64).ravel()
    tb = np.asarray(Image.fromarray(b).resize((16, 16), Image.LANCZOS),
                    dtype=np.float64).ravel()
    ca = np.corrcoef(ta, tb)[0, 1]
    return ca if ca == ca else 0.0


def attention_select_slices(volume, k):
    """Attention-based key-slice selection.

    Each slice gets a saliency score (gradient/edge energy x entropy x
    non-background content x a weak center-of-volume prior), then a greedy,
    diversity-aware pass picks the *k* slices: at each step the best slice is
    the one with the highest saliency not already represented by an already
    selected slice (measured by normalized cross-correlation). This mimics
    attention: it favors informative slices and discards near-duplicate ones.
    Returns the selected z-indices in ascending z order.
    """
    n = volume.shape[0]
    k = max(1, min(int(k), n))
    if n <= k:
        return list(range(n))

    mid = (n - 1) / 2.0
    center_prior = [np.exp(-0.5 * ((z - mid) / (0.35 * n)) ** 2)
                    for z in range(n)]
    scores = np.array([
        (_slice_edge_energy(volume[z]) * (0.5 + _slice_entropy(volume[z]))
         * (0.3 + _slice_non_background(volume[z]))
         * (0.5 + center_prior[z]))
        for z in range(n)], dtype=np.float64)
    if not np.isfinite(scores).all() or scores.max() <= 0:
        scores = np.arange(n, dtype=np.float64) + 1.0

    selected = []
    candidates = list(range(n))
    while len(selected) < k and candidates:
        cur = scores[candidates].copy()
        if selected:
            for i, cand in enumerate(candidates):
                sim = max(abs(_slice_similarity(volume[cand], volume[j]))
                          for j in selected)
                cur[i] *= max(0.2, 1.0 - sim)
        pick = candidates[int(np.argmax(cur))]
        selected.append(pick)
        candidates.remove(pick)
    return sorted(selected)


def _window_slice_uint8(arr, modality):
    """Fixed-window a raw slice to uint8 (removes DICOM intensity calibration)."""
    arr = np.asarray(arr, dtype=np.float64)
    if modality == "CT":
        return _apply_ct_window(arr, 40.0, 400.0)
    return _stretch_to_uint8(arr)


def _best_effort_affine(frames, n_z):
    """Build an approximate DICOM-like affine (mm) for the NIfTI header."""
    spacing = [1.0, 1.0, 1.0]
    for f in frames:
        if f.get("spacing") and len(f["spacing"]) >= 2:
            spacing[0], spacing[1] = f["spacing"][0], f["spacing"][1]
            break
    for f in frames:
        loc = f.get("slice_loc")
        if np.isfinite(loc):
            zs = sorted({round(loc, 3) for loc in
                         (fr.get("slice_loc") for fr in frames)
                         if np.isfinite(loc)})
            if len(zs) >= 2:
                spacing[2] = abs(zs[1] - zs[0])
            break
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return affine


def write_nifti(image, out_path, affine=None):
    """Write a volume (z,y,x) to a NIfTI-1 file."""
    import nibabel as nib
    affine = np.eye(4) if affine is None else affine
    vol = np.asarray(image)
    if vol.dtype != np.float32 and vol.dtype != np.int16:
        vol = vol.astype(np.float32)
    nib.save(nib.Nifti1Image(vol, affine), str(out_path))
    return out_path


def process_upload_dicom(paths, max_slices=MAX_PROMPT_IMAGES, dest_dir=None,
                         write_volume=True):
    """Parse a DICOM series into a cleaned NIfTI volume and windowed previews.

    1. Every frame is read raw and assembled into a z-stacked volume.
    2. Intensity calibration is removed: CT uses a fixed anatomical window,
       everything else is percentile-stretched; near-constant, empty slices
       (air-only/padding) are deleted.
    3. The cleaned volume is saved as NIfTI (when ``dest_dir`` is given).
    4. Key slices are chosen with attention-based selection and windowed to
       PNGs; the original DICOM instance numbers are returned as references.
    """
    import tempfile

    dicom_files = collect_dicom_files(paths)
    if not dicom_files:
        raise ValueError("No DICOM files (.dcm) found in the uploaded files.")
    if len(dicom_files) > 60:
        logger.warning("Large DICOM series (%d files); sampling for prompt.",
                       len(dicom_files))

    frames = []
    skipped_files = []
    for f in dicom_files:
        try:
            frames.extend(_read_dicom_frames(f))
        except Exception as e:  # noqa: BLE001 - skip one bad file, keep series
            logger.warning("Skipping unreadable DICOM file %s: %s", f, e)
            skipped_files.append(f"{f.name} ({e})")
    if not frames:
        raise ValueError("No readable DICOM slices found in the uploaded files.")
    frames = _sort_frames(frames)

    modalities = {f["modality"] for f in frames}
    modality = "CT" if "CT" in modalities else (
               "MRI" if "MRI" in modalities else "X-ray")
    is_ct = modality == "CT"

    # Straighten to HU for CT (needed for the fixed window), otherwise raw.
    raw = []
    for f in frames:
        a = np.asarray(f["array"], dtype=np.float64)
        if is_ct:
            a = a * f["slope"] + f["intercept"]
        raw.append(a)
    try:
        volume = np.stack(raw, axis=0)
    except ValueError:
        # Frames of unequal shape: keep the common grid, fall back to
        # nearest-neighbor resampling for outliers.
        h, w = raw[0].shape
        rs = [np.resize(a, (h, w)) for a in raw]
        volume = np.stack(rs, axis=0)

    keep_z = _mask_empty_slices(volume)
    volume = volume[keep_z]
    kept_frames = [frames[z] for z in keep_z]
    total = volume.shape[0]

    cleaned = _decalibrate_volume(volume, is_ct)

    nifti_path = None
    if write_volume and dest_dir is not None:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / "volume_cleaned.nii.gz"
        try:
            affine = _best_effort_affine(kept_frames, total)
            nifti_array = (np.round(volume).astype(np.int16) if is_ct
                           else volume.astype(np.float32))
            write_nifti(nifti_array, out_path, affine=affine)
            nifti_path = str(out_path)
        except Exception as e:  # noqa: BLE001 - NIfTI is a nicety
            logger.warning("Could not write NIfTI volume: %s", e)

    selected_z = attention_select_slices(cleaned, max_slices)
    previews = []
    slice_references = []
    for i, z in enumerate(selected_z, 1):
        pixels = _window_slice_uint8(volume[z], modality)
        inst = int(kept_frames[z]["instance"])
        previews.append({
            "label": f"Slice {z + 1}/{total} (inst {inst})",
            "data_url": array_to_png_data_url(pixels),
        })
        slice_references.append(inst)

    return {
        "modality": modality,
        "total_slices": total,
        "prompt_slices": len(previews),
        "previews": previews,
        "slice_references": slice_references,
        "series_desc": kept_frames[0]["series_desc"],
        "body_part": kept_frames[0]["body_part"],
        "nifti_path": nifti_path,
        "skipped_files": skipped_files[:20],
    }


def process_upload(paths, max_slices=MAX_PROMPT_IMAGES, modality_override=None,
                   dest_dir=None):
    """Unified entry point for /upload_explain.

    Splits the uploaded files into DICOM (.dcm/.dicom/.zip) and plain-image
    (PNG/JPG/...) parts. DICOM tags drive auto-detection when present (they
    are authoritative); plain-image-only uploads default to X-ray. Files that
    cannot be parsed are reported in ``skipped_files`` instead of silently
    shrinking the series.
    """
    plain_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    dicom_paths, plain_paths = [], []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() in plain_exts:
            plain_paths.append(p)
        else:
            dicom_paths.append(p)

    skipped = []
    if not dicom_paths:
        result = process_plain_images(plain_paths, max_slices=max_slices)
        skipped = result["skipped_files"]
        result["modality"] = modality_override or result["modality"]
    else:
        dcm_result = process_upload_dicom(dicom_paths, max_slices=max_slices,
                                          dest_dir=dest_dir)
        result = dict(dcm_result)
        skipped = list(dcm_result["skipped_files"])
        if plain_paths:
            # Merge plain images in as extra slices of the same study so no
            # file the user picked is silently dropped.
            plain_result = process_plain_images(
                plain_paths, max_slices=max(1, max_slices - len(result["previews"])))
            for prev in plain_result["previews"]:
                result["previews"].append(prev)
            skipped += plain_result["skipped_files"]
            result["total_slices"] = dcm_result["total_slices"] + \
                plain_result["total_slices"]
            result["prompt_slices"] = len(result["previews"])
        if modality_override:
            result["modality"] = modality_override
    result["skipped_files"] = skipped[:20]
    return result


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
    min_instances = IDC_MIN_INSTANCES.get(modality, 1)
    select_cols = """
SELECT i.SeriesInstanceUID AS SeriesInstanceUID,
       i.collection_id AS collection_id,
       i.SeriesDescription AS SeriesDescription,
       i.BodyPartExamined AS BodyPartExamined,
       i.series_size_MB AS series_size_MB,
       (SELECT count(*) FROM index b
        WHERE b.SeriesInstanceUID = i.SeriesInstanceUID) AS n_instances
FROM index i
"""
    # IMPORTANT: no upper size ceiling in SQL. Most real CT series in IDC are
    # larger than 80 MB, so a hard `series_size_MB < 80` filter turned the
    # query (and its fallback, which reused the same filter) empty and
    # produced "No public CT series ... was found". Instead we ask for the
    # smallest series of the modality and pick the best candidate in Python.
    where_frags = [f"i.Modality IN ({in_mods})",
                   f"i.series_size_MB > {IDC_SIZE_MB_MIN}",
                   scout_filters]
    query = select_cols + ("WHERE " + " AND ".join(where_frags) + f"""
ORDER BY CASE WHEN i.BodyPartExamined IN ({in_body}) THEN 0 ELSE 1 END,
         i.series_size_MB ASC
LIMIT 300
""")
    rows = client.sql_query(query)
    rows = None if rows is None or len(rows) == 0 else rows
    if rows is None:
        # Fallback query: same selection, but ordered by instance count so we
        # expose only the smallest real series if the subquery-based ordering
        # is not supported by the underlying engine.
        rows = client.sql_query(
            select_cols + ("WHERE " + " AND ".join(where_frags) + f"""
ORDER BY n_instances ASC, series_size_MB ASC
LIMIT 300
"""))
        rows = None if rows is None or len(rows) == 0 else rows
    if rows is None:
        raise ValueError(
            f"No public {modality} series with at least {min_instances} "
            f"instances was found in IDC for this query.")

    records = rows.to_dict("records")

    def _inst(r):
        return int(r.get("n_instances") or 0)

    def _size(r):
        return float(r.get("series_size_MB") or 0)

    def _pref(r):
        body = (r.get("BodyPartExamined") or "").strip().upper()
        return 0 if body in preferred else 1

    # Only series with the requested minimum number of slices qualify. Sort by
    # smallest size first (preferred body part as a tiebreaker) so the demo
    # downloads fast and honors the chosen Slice count.
    candidates = [r for r in records if _inst(r) >= min_instances]
    candidates.sort(key=lambda r: (_size(r), _pref(r), _inst(r)))
    if not candidates:
        raise ValueError(
            f"No public {modality} series with at least {min_instances} "
            f"instances was found in IDC for this query.")

    infos = []
    for r in candidates:
        instances = _inst(r)
        infos.append({
            "series_uid": r["SeriesInstanceUID"],
            "collection_id": r.get("collection_id") or "",
            "series_description": r.get("SeriesDescription") or "",
            "body_part": r.get("BodyPartExamined") or "",
            "size_mb": _size(r),
            "instances": instances,
            # True when the pick falls outside the preferred "small demo
            # series" band ([:8, 400] slices and <= IDC_SIZE_MB_MAX).
            "is_fallback": (instances > IDC_MAX_INSTANCES
                            or _size(r) > IDC_SIZE_MB_MAX),
        })
        if len(infos) >= limit:
            break
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