"""
Layout-Aware Text Extraction Pipeline v4 — Dual OCR

Arsitektur dua pipeline OCR:

  Pipeline 1  (paragraph=True)
    → Merge baris-baris menjadi satu blok teks yang rapi dan lengkap
    → Digunakan sebagai isi <span> contenteditable di HTML
    → Posisi (bbox) dari pipeline ini dipakai untuk overlay

  Pipeline 2  (paragraph=False)
    → Satu bbox per baris teks → bbox jauh lebih kecil dan presisi
    → Digunakan untuk DUA hal di belakang layar:
        a) Font size: median tinggi bbox-per-baris = estimasi font size 1 baris (akurat)
        b) Inpainting mask: bbox kecil per baris → mask tidak "banjiri" area non-teks
           (penting untuk page 4 dengan background segi enam)

Keduanya berbagi preprocessing yang sama (mask logo, resize, bilateral, thresh, median).
"""

import math
import numpy as np
import easyocr
import cv2
import os
import base64
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCALE       = 2              # resize factor preprocessing OCR
LOGO_MASK   = (4700, 0, None, 700)  # (x1,y1,x2,y2) area logo yang di-mask
DISPLAY_W   = 1280           # lebar slide HTML (px)
INPAINT_R   = 8              # radius inpainting (lebih kecil = lebih presisi pada hexagon)
INPAINT_PAD = 3              # padding mask per bbox pipeline-2 (px, ruang gambar asli)

# Estimasi lebar karakter relatif terhadap font_size (Segoe UI 600-weight, campuran huruf)
# Digunakan untuk memastikan single-line text tidak wrap di HTML
CHAR_W_RATIO = 0.55

reader = easyocr.Reader(['id', 'en'], gpu=True)


# ─────────────────────────────────────────────
# PREPROCESSING (bersama)
# ─────────────────────────────────────────────
def preprocess(image_path):
    """
    Muat gambar, mask logo, resize, dan jalankan seluruh preprocessing.
    Return: img_original (BGR), preprocessed_gray (uint8), orig_w, orig_h
    """
    img_original = cv2.imread(str(image_path))
    if img_original is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    orig_h, orig_w = img_original.shape[:2]

    img = img_original.copy()
    x2_logo = LOGO_MASK[2] if LOGO_MASK[2] else orig_w
    cv2.rectangle(img, (LOGO_MASK[0], LOGO_MASK[1]), (x2_logo, LOGO_MASK[3]), (255, 0, 0), -1)

    img_resized = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    gray      = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 100, 100)
    thresh    = cv2.threshold(bilateral, 90, 255, cv2.THRESH_BINARY)[1]
    median    = cv2.medianBlur(thresh, 7)

    return img_original, median, orig_w, orig_h


# ─────────────────────────────────────────────
# HELPER: CLEAN TEXT
# ─────────────────────────────────────────────
def clean_text(text):
    t = text.strip()
    t = t.replace(";", ",").replace("'", ",")
    t = t.replace(" ,", ",").replace(" .", ".")
    if t.endswith(","):
        t = t[:-1] + "."
    return t


# ─────────────────────────────────────────────
# PIPELINE 1 — paragraph=True (untuk HTML editable text)
# ─────────────────────────────────────────────
def run_pipeline1(preprocessed, img_original):
    """
    OCR paragraph=True: merge baris menjadi blok teks lengkap.
    Return: list of dict {text, x, y, width, height, r, g, b}
    Koordinat dalam ruang gambar ASLI.
    """
    results = reader.readtext(
        preprocessed,
        paragraph=True,
        x_ths=1.6,
        y_ths=0.2,
        width_ths=0.7,
    )

    elements = []
    for (bbox, text) in results:
        t = clean_text(text)
        if len(t) <= 2:
            continue

        x1 = int(bbox[0][0] / SCALE)
        y1 = int(bbox[0][1] / SCALE)
        x2 = int(bbox[2][0] / SCALE)
        y2 = int(bbox[2][1] / SCALE)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        r, g, b = get_text_color(img_original, x1, y1, x2, y2)

        elements.append({
            "text":   t,
            "x": x1, "y": y1,
            "width": box_w, "height": box_h,
            "r": r, "g": g, "b": b,
        })

    return elements


# ─────────────────────────────────────────────
# PIPELINE 2 — paragraph=False (untuk font size & inpainting)
# ─────────────────────────────────────────────
def run_pipeline2(preprocessed):
    """
    OCR paragraph=False: satu bbox per baris/kata.
    Return: list of dict {x, y, width, height}
    Koordinat dalam ruang gambar ASLI.
    Teks tidak dipakai, hanya geometri bbox.
    """
    results = reader.readtext(
        preprocessed,
        paragraph=False,
    )

    line_boxes = []
    for (bbox, text, conf) in results:
        t = clean_text(text)
        if len(t) <= 1 or conf < 0.3:
            continue

        x1 = int(bbox[0][0] / SCALE)
        y1 = int(bbox[0][1] / SCALE)
        x2 = int(bbox[2][0] / SCALE)
        y2 = int(bbox[2][1] / SCALE)
        box_h = max(1, y2 - y1)

        line_boxes.append({
            "x": x1, "y": y1,
            "width": max(1, x2 - x1),
            "height": box_h,
        })

    return line_boxes


# ─────────────────────────────────────────────
# HELPER: WARNA TEKS
# ─────────────────────────────────────────────
def get_text_color(img_bgr, x1, y1, x2, y2):
    """K-Means 2-cluster pada ROI → warna teks (cluster minoritas)."""
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return (30, 30, 30)

    roi = img_bgr[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    if len(roi) < 4:
        return (30, 30, 30)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(roi, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    text_cluster = np.argmin(counts)
    b, g, r = centers[text_cluster].astype(int)
    return (int(r), int(g), int(b))


# ─────────────────────────────────────────────
# FONT SIZE — dari pipeline 2 (presisi per baris)
# ─────────────────────────────────────────────
def compute_font_sizes(elements_p1, line_boxes_p2, scale_factor):
    """
    Untuk setiap elemen paragraph (pipeline 1), cari bbox baris (pipeline 2)
    yang berada di dalam bbox paragraph tersebut, ambil MEDIAN tinggi baris
    → font size presisi untuk 1 baris.

    Jika tidak ada bbox-p2 yang cocok (fallback), pakai formula sqrt dari v3.

    Parameter:
        elements_p1  : hasil pipeline 1 (koordinat ruang asli)
        line_boxes_p2: hasil pipeline 2 (koordinat ruang asli)
        scale_factor : orig → display

    Return: list font_sizes (px, sudah di-scale ke display), urutan sama dengan elements_p1.
    """
    FONT_RATIO = 0.80   # median_line_h × ratio = font_size
    # Fallback formula parameter (v3)
    CHAR_W = 0.74
    LINE_H = 1.55

    font_sizes = []
    for el in elements_p1:
        # Kumpulkan bbox-p2 yang centre-nya berada dalam bbox-p1
        el_cx_min = el["x"]
        el_cx_max = el["x"] + el["width"]
        el_cy_min = el["y"]
        el_cy_max = el["y"] + el["height"]

        matching_heights = []
        for lb in line_boxes_p2:
            cx = lb["x"] + lb["width"]  / 2
            cy = lb["y"] + lb["height"] / 2
            if el_cx_min <= cx <= el_cx_max and el_cy_min <= cy <= el_cy_max:
                matching_heights.append(lb["height"])

        if matching_heights:
            # Median tinggi baris → font size 1 baris (presisi)
            median_h   = float(np.median(matching_heights))
            font_px_orig = median_h * FONT_RATIO
        else:
            # Fallback: formula sqrt dari v3
            w_sc = int(el["width"]  * scale_factor)
            h_sc = int(el["height"] * scale_factor)
            n    = max(1, len(el["text"]))
            font_px_orig = math.sqrt(w_sc * h_sc / (n * CHAR_W * LINE_H))
            font_px_orig = min(font_px_orig, h_sc * 0.80)

        # Scale ke display
        font_px = max(8, int(font_px_orig * scale_factor))
        font_sizes.append(font_px)

    return font_sizes


# ─────────────────────────────────────────────
# INPAINTING — dari pipeline 2 (mask presisi per baris)
# ─────────────────────────────────────────────
def inpaint_text(img_bgr, line_boxes_p2):
    """
    Buat mask dari bbox PIPELINE 2 (per baris, bukan per paragraf).
    → Mask jauh lebih kecil dan mengikuti bentuk teks yang sebenarnya,
      tidak membanjiri area non-teks seperti background segi enam (page 4).

    Mask di-dilate sedikit untuk menangkap antialiasing tepi teks.
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    h_img, w_img = img_bgr.shape[:2]

    for lb in line_boxes_p2:
        pad = INPAINT_PAD
        x1 = max(0, lb["x"] - pad)
        y1 = max(0, lb["y"] - pad)
        x2 = min(w_img, lb["x"] + lb["width"]  + pad)
        y2 = min(h_img, lb["y"] + lb["height"] + pad)
        mask[y1:y2, x1:x2] = 255

    kernel   = np.ones((3, 3), np.uint8)
    mask     = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(img_bgr, mask, INPAINT_R, cv2.INPAINT_TELEA)
    return inpainted


# ─────────────────────────────────────────────
# WIDTH & ALIGNMENT — dari pipeline 2
# ─────────────────────────────────────────────
def _overlapping_p2(el, line_boxes_p2, slack=30):
    """
    Kembalikan list bbox P2 yang pusat-nya masuk dalam bbox P1.
    slack (px, ruang asli): toleransi horizontal agar bbox 1-baris
    yang sedikit melampaui P1 tetap tertangkap.
    """
    el_x1, el_y1 = el["x"], el["y"]
    el_x2, el_y2 = el["x"] + el["width"], el["y"] + el["height"]
    matched = []
    for lb in line_boxes_p2:
        cx = lb["x"] + lb["width"]  / 2
        cy = lb["y"] + lb["height"] / 2
        if (el_x1 - slack <= cx <= el_x2 + slack) and (el_y1 <= cy <= el_y2):
            matched.append(lb)
    return matched


def refine_width_from_p2(el, line_boxes_p2):
    """
    Perluas lebar bbox P1 menggunakan batas kanan (max x2) dari baris P2
    yang overlap. Width hanya bisa bertambah, tidak pernah berkurang.
    Return: refined_width (int, ruang gambar asli).
    """
    el_x2 = el["x"] + el["width"]
    max_x2 = el_x2
    for lb in _overlapping_p2(el, line_boxes_p2):
        max_x2 = max(max_x2, lb["x"] + lb["width"])
    return max(el["width"], max_x2 - el["x"])


def detect_alignment(el, line_boxes_p2):
    """
    Deteksi text-align elemen secara otomatis menggunakan distribusi baris P2.

    STRATEGI
    ─────────
    Single-line (< 2 baris P2 overlap):
      Kembalikan 'left'. Width sudah di-fit ke panjang teks (lihat generate_html),
      sehingga left vs center tidak membuat perbedaan visual. 'left' lebih aman
      untuk judul yang memang rata kiri.

    Multi-line (≥ 2 baris P2 overlap):
      Hitung variance dari tiga metrik posisi setiap baris P2:
        • var_left   = variance(x_left  tiap baris)   → kecil jika rata-kiri
        • var_center = variance(x_center tiap baris)  → kecil jika rata-tengah
        • var_right  = variance(x_right  tiap baris)  → kecil jika rata-kanan
      Alignment dengan variance TERKECIL = alignment sesungguhnya.

      Contoh:
        "Filosofi Sarang Lebah..." (page-0004, center):
          Setiap baris P2 mempunyai center mendekati pusat hexagon
          → var_center terkecil → 'center' ✓

        "Program SEAL, Talenesia..." (page-0003, center):
          Tiap baris rata tengah di dalam card
          → var_center terkecil → 'center' ✓

        "Mari Memungkinkan Transformasi..." (page-0001, left):
          Tiap baris mulai dari x yang hampir sama
          → var_left terkecil → 'left' ✓

    Return: 'left' | 'center' | 'right'
    """
    lines = _overlapping_p2(el, line_boxes_p2, slack=0)  # ketat untuk alignment

    if len(lines) < 2:
        return "left"

    lefts   = [lb["x"]                          for lb in lines]
    centers = [lb["x"] + lb["width"] / 2        for lb in lines]
    rights  = [lb["x"] + lb["width"]            for lb in lines]

    var_l = float(np.var(lefts))
    var_c = float(np.var(centers))
    var_r = float(np.var(rights))

    if min(var_l, var_c, var_r) == var_c:
        return "center"
    if min(var_l, var_c, var_r) == var_r:
        return "right"
    return "left"


# ─────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────
def img_to_b64(img_bgr):
    """Encode gambar OpenCV ke base64 JPEG (lebih kecil dari PNG untuk foto)."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    _, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def generate_html(image_path, elements_p1, font_sizes, line_boxes_p2,
                  orig_w, orig_h, output_path):
    """
    Buat HTML overlay:
    - Background: gambar hasil inpainting (mask dari pipeline 2, presisi per baris)
    - Overlay: <span contenteditable> per elemen pipeline 1
    - Font size: dari pipeline 2 (median tinggi baris)
    - Posisi: dari pipeline 1 (bbox paragraf)
    """
    scale_factor = DISPLAY_W / orig_w
    display_h    = int(orig_h * scale_factor)

    # ── Inpainting menggunakan bbox pipeline 2 ──
    print("  → Inpainting (mask presisi pipeline-2) ...")
    img_bgr   = cv2.imread(str(image_path))
    img_clean = inpaint_text(img_bgr, line_boxes_p2)
    bg_b64    = img_to_b64(img_clean)

    # ── Bangun <span> dari pipeline 1, font size dari pipeline 2 ──
    spans = []
    for el, font_px in zip(elements_p1, font_sizes):
        left   = int(el["x"]      * scale_factor)
        top    = int(el["y"]      * scale_factor)
        height = int(el["height"] * scale_factor)
        color  = f"rgb({el['r']},{el['g']},{el['b']})"
        safe   = (el["text"]
                  .replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;"))

        # ── Lebar ──────────────────────────────────────────────────────────
        # Langkah 1: perluas dengan max(x2) dari baris P2 yang overlap
        refined_w_orig = refine_width_from_p2(el, line_boxes_p2)
        width = int(refined_w_orig * scale_factor)

        # Langkah 2 (single-line): pastikan lebar cukup untuk 1 baris penuh.
        # Untuk teks 1 baris, P2 juga bisa sempit karena OCR mengikuti piksel
        # teks yang ketat. Gunakan estimasi karakter agar tidak wrap di browser.
        n_p2 = len(_overlapping_p2(el, line_boxes_p2))
        if n_p2 <= 1:
            min_w_single = int(len(el["text"]) * font_px * CHAR_W_RATIO)
            width = max(width, min_w_single)

        # ── Alignment ──────────────────────────────────────────────────────
        # Single-line → 'left' (width sudah fit ke teks, tidak ada efek visual)
        # Multi-line  → deteksi otomatis dari variance posisi baris P2
        alignment = detect_alignment(el, line_boxes_p2)

        style = (
            f"left:{left}px;top:{top}px;"
            f"width:{width}px;min-height:{height}px;"
            f"font-size:{font_px}px;color:{color};"
            f"text-align:{alignment};"
        )
        spans.append(
            f'  <span class="txt" style="{style}" contenteditable="true">{safe}</span>'
        )

    spans_html = "\n".join(spans)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>{Path(output_path).stem}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a1a; display: flex; justify-content: center; padding: 20px; }}
  .slide {{
    position: relative;
    width: {DISPLAY_W}px;
    height: {display_h}px;
    flex-shrink: 0;
  }}
  .slide img.bg {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    display: block;
  }}
  .txt {{
    position: absolute;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-weight: 600;
    line-height: 1.2;
    white-space: pre-wrap;
    word-break: break-word;
    overflow: visible;
    cursor: text;
    outline: none;
    z-index: 10;
    background-color: transparent;
    padding: 0 2px;
  }}
  .txt:hover {{ outline: 1px dashed rgba(0,120,255,0.4); background: rgba(255,255,255,0.08); }}
  .txt:focus {{ outline: 2px solid #0078ff; background: rgba(255,255,255,0.92); color: #111 !important; }}
</style>
</head>
<body>
<div class="slide">
  <img class="bg" src="{bg_b64}" alt="background">
{spans_html}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Disimpan: {output_path}")


# ─────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────
def process_slide(image_path, output_dir="."):
    image_path  = Path(image_path)
    output_path = Path(output_dir) / (image_path.stem + ".html")

    print(f"\n{'='*60}")
    print(f"  {image_path.name}")
    print(f"{'='*60}")

    # 1. Preprocessing bersama (1×)
    img_original, preprocessed, orig_w, orig_h = preprocess(image_path)
    scale_factor = DISPLAY_W / orig_w

    # 2. Pipeline 1 — paragraph=True (teks editable)
    print("  [P1] OCR paragraph=True ...")
    elements_p1 = run_pipeline1(preprocessed, img_original)
    print(f"       {len(elements_p1)} blok teks ditemukan")

    # 3. Pipeline 2 — paragraph=False (bbox presisi per baris)
    print("  [P2] OCR paragraph=False ...")
    line_boxes_p2 = run_pipeline2(preprocessed)
    print(f"       {len(line_boxes_p2)} baris/bbox ditemukan")

    # 4. Hitung font size dari pipeline 2
    font_sizes = compute_font_sizes(elements_p1, line_boxes_p2, scale_factor)

    # 5. Debug log
    for el, fs in zip(elements_p1, font_sizes):
        n_p2      = len(_overlapping_p2(el, line_boxes_p2))
        align     = detect_alignment(el, line_boxes_p2)
        refined_w = refine_width_from_p2(el, line_boxes_p2)
        disp_w    = int(refined_w * scale_factor)
        if n_p2 <= 1:
            disp_w = max(disp_w, int(len(el["text"]) * fs * CHAR_W_RATIO))
        print(f"    [{int(el['x']*scale_factor):4},{int(el['y']*scale_factor):4}] "
              f"w={disp_w:4}px fs={fs:2}px p2={n_p2} align={align:<6}  →  {el['text'][:55]}")

    # 6. Generate HTML (inpainting + overlay)
    generate_html(image_path, elements_p1, font_sizes, line_boxes_p2,
                  orig_w, orig_h, output_path)
    return str(output_path)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR   = "task_b_layout_extraction/data"
    OUTPUT_DIR = "task_b_layout_extraction/output_html_ide4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(1, 6):
        fname = (
            f"Profile Image Studio  Let's Enable Digital Transformation "
            f"with Us (2)_page-000{i}.jpg"
        )
        img_path = Path(DATA_DIR) / fname
        if img_path.exists():
            process_slide(img_path, OUTPUT_DIR)
        else:
            print(f"[skip] tidak ditemukan: {img_path}")

    print("\n✅ Semua slide selesai!")