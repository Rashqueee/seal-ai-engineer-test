import math
import numpy as np
import easyocr
import cv2
import os
import base64
from pathlib import Path

reader = easyocr.Reader(['id', 'en'], gpu=True)


# PREPROCESSING
def preprocess(image_path):
    # Load gambar, mask logo, resize, dan jalankan keseluruhan preprocessing OCR
    img_original = cv2.imread(str(image_path))
    if img_original is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    orig_h, orig_w = img_original.shape[:2]
    img = img_original.copy()
    
    # Masking area logo (x1=4700, y1=0, x2=orig_w, y2=700)
    cv2.rectangle(img, (4700, 0), (orig_w, 700), (255, 0, 0), -1)

    # Resize factor 2x
    img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray      = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 100, 100)
    thresh    = cv2.threshold(bilateral, 90, 255, cv2.THRESH_BINARY)[1]
    median    = cv2.medianBlur(thresh, 7)

    return img_original, median, orig_w, orig_h


# CLEAN TEXT
def clean_text(text):
    # Post-processing teks OCR untuk memperbaiki tanda baca dan menghapus spasi berlebih
    t = text.strip()
    t = t.replace(";", ",").replace("'", ",")
    t = t.replace(" ,", ",").replace(" .", ".")
    if t.endswith(","):
        t = t[:-1] + "."
    return t


# PIPELINE 1: paragraph=True
def pipeline1(preprocessed, img_original):
    # OCR paragraph=True untuk merge baris menjadi blok teks lengkap beserta koordinatnya
    results = reader.readtext(
        preprocessed, paragraph=True, x_ths=1.6, y_ths=0.2, width_ths=0.7
    )

    elements = []
    for (bbox, text) in results:
        t = clean_text(text)
        if len(t) <= 2:
            continue

        # Koordinat di-scale kembali (dibagi 2) ke ukuran original
        x1, y1 = int(bbox[0][0] / 2), int(bbox[0][1] / 2)
        x2, y2 = int(bbox[2][0] / 2), int(bbox[2][1] / 2)
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        r, g, b = get_text_color(img_original, x1, y1, x2, y2)

        elements.append({
            "text": t, "x": x1, "y": y1,
            "width": box_w, "height": box_h,
            "r": r, "g": g, "b": b,
        })

    return elements


# PIPELINE 2: paragraph=False
def pipeline2(preprocessed):
    # OCR paragraph=False untuk mendapatkan geometri bbox presisi per baris
    results = reader.readtext(preprocessed, paragraph=False)

    line_boxes = []
    for (bbox, text, conf) in results:
        t = clean_text(text)
        if len(t) <= 1 or conf < 0.3:
            continue

        # Koordinat di-scale kembali (dibagi 2) ke ukuran original
        x1, y1 = int(bbox[0][0] / 2), int(bbox[0][1] / 2)
        x2, y2 = int(bbox[2][0] / 2), int(bbox[2][1] / 2)
        box_h = max(1, y2 - y1)

        line_boxes.append({
            "x": x1, "y": y1, "width": max(1, x2 - x1), "height": box_h,
        })

    return line_boxes


# WARNA TEKS
def get_text_color(img_bgr, x1, y1, x2, y2):
    # Mengekstrak warna teks (cluster minoritas) dari ROI gambar menggunakan K-Means
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
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


# FONT SIZE
def compute_font_sizes(elements_p1, line_boxes_p2, scale_factor):
    # Menghitung ukuran font dari median tinggi bbox baris P2 di dalam area P1
    FONT_RATIO = 0.80
    CHAR_W, LINE_H = 0.74, 1.55

    font_sizes = []
    for el in elements_p1:
        el_cx_min, el_cx_max = el["x"], el["x"] + el["width"]
        el_cy_min, el_cy_max = el["y"], el["y"] + el["height"]

        matching_heights = []
        for lb in line_boxes_p2:
            cx, cy = lb["x"] + lb["width"] / 2, lb["y"] + lb["height"] / 2
            if el_cx_min <= cx <= el_cx_max and el_cy_min <= cy <= el_cy_max:
                matching_heights.append(lb["height"])

        if matching_heights:
            median_h = float(np.median(matching_heights))
            font_px_orig = median_h * FONT_RATIO
        else:
            w_sc, h_sc = int(el["width"] * scale_factor), int(el["height"] * scale_factor)
            n = max(1, len(el["text"]))
            font_px_orig = math.sqrt(w_sc * h_sc / (n * CHAR_W * LINE_H))
            font_px_orig = min(font_px_orig, h_sc * 0.80)

        font_px = max(8, int(font_px_orig * scale_factor))
        font_sizes.append(font_px)

    return font_sizes


# INPAINTING
def inpaint_text(img_bgr, line_boxes_p2):
    # Membuat inpainting mask dari bbox P2 untuk menghapus teks pada background secara presisi
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    h_img, w_img = img_bgr.shape[:2]

    for lb in line_boxes_p2:
        # Menggunakan padding inpainting sebesar 3px
        pad = 3
        x1, y1 = max(0, lb["x"] - pad), max(0, lb["y"] - pad)
        x2, y2 = min(w_img, lb["x"] + lb["width"] + pad), min(h_img, lb["y"] + lb["height"] + pad)
        mask[y1:y2, x1:x2] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Menjalankan inpainting dengan radius 8px
    return cv2.inpaint(img_bgr, mask, 8, cv2.INPAINT_TELEA)


# WIDTH & ALIGNMENT
def overlapping_p2(el, line_boxes_p2, slack=30):
    # Mendapatkan daftar bbox P2 yang pusatnya berada di dalam rentang bbox P1
    el_x1, el_y1 = el["x"], el["y"]
    el_x2, el_y2 = el["x"] + el["width"], el["y"] + el["height"]
    matched = []
    
    for lb in line_boxes_p2:
        cx, cy = lb["x"] + lb["width"] / 2, lb["y"] + lb["height"] / 2
        if (el_x1 - slack <= cx <= el_x2 + slack) and (el_y1 <= cy <= el_y2):
            matched.append(lb)
            
    return matched

def refine_width_from_p2(el, line_boxes_p2):
    # Menyesuaikan dan memperluas lebar bbox P1 berdasarkan batas kanan maksimal baris P2
    max_x2 = el["x"] + el["width"]
    for lb in overlapping_p2(el, line_boxes_p2):
        max_x2 = max(max_x2, lb["x"] + lb["width"])
    return max(el["width"], max_x2 - el["x"])

def detect_alignment(el, line_boxes_p2):
    # Mendeteksi text-align secara otomatis (left, center, right) menggunakan variance
    lines = overlapping_p2(el, line_boxes_p2, slack=0)
    if len(lines) < 2:
        return "left"

    lefts   = [lb["x"] for lb in lines]
    centers = [lb["x"] + lb["width"] / 2 for lb in lines]
    rights  = [lb["x"] + lb["width"] for lb in lines]

    var_l, var_c, var_r = float(np.var(lefts)), float(np.var(centers)), float(np.var(rights))

    if min(var_l, var_c, var_r) == var_c:
        return "center"
    if min(var_l, var_c, var_r) == var_r:
        return "right"
    return "left"


# HTML GENERATION
def img_to_b64(img_bgr):
    # Menerjemahkan format citra OpenCV menjadi base64 JPEG format
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    _, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def generate_html(image_path, elements_p1, font_sizes, line_boxes_p2,
                  orig_w, orig_h, output_path):
    # Menyusun struktur HTML overlay
    # Menggunakan lebar display tetap 1280px
    scale_factor = 1280 / orig_w
    display_h    = int(orig_h * scale_factor)

    img_bgr   = cv2.imread(str(image_path))
    img_clean = inpaint_text(img_bgr, line_boxes_p2)
    bg_b64    = img_to_b64(img_clean)

    spans = []
    for el, font_px in zip(elements_p1, font_sizes):
        left   = int(el["x"] * scale_factor)
        top    = int(el["y"] * scale_factor)
        height = int(el["height"] * scale_factor)
        color  = f"rgb({el['r']},{el['g']},{el['b']})"
        safe   = (el["text"]
                  .replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;"))

        refined_w_orig = refine_width_from_p2(el, line_boxes_p2)
        width = int(refined_w_orig * scale_factor)

        n_p2 = len(overlapping_p2(el, line_boxes_p2))
        if n_p2 <= 1:
            # Menggunakan estimasi lebar karakter ratio 0.55
            min_w_single = int(len(el["text"]) * font_px * 0.55)
            width = max(width, min_w_single)

        alignment = detect_alignment(el, line_boxes_p2)

        style = (
            f"left:{left}px;top:{top}px;"
            f"width:{width}px;min-height:{height}px;"
            f"font-size:{font_px}px;color:{color};"
            f"text-align:{alignment};"
        )
        spans.append(
            f'        <span class="txt" style="{style}" contenteditable="true">{safe}</span>'
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
            width: 1280px;
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


# PIPELINE UTAMA
def process_slide(image_path, output_dir="."):
    # Mengendalikan seluruh alur eksekusi pipeline untuk memproses satu gambar presentasi
    image_path  = Path(image_path)
    output_path = Path(output_dir) / (image_path.stem + ".html")

    img_original, preprocessed, orig_w, orig_h = preprocess(image_path)
    scale_factor = 1280 / orig_w # Menggunakan skala relatif terhadap 1280px

    elements_p1 = pipeline1(preprocessed, img_original)
    line_boxes_p2 = pipeline2(preprocessed)
    font_sizes = compute_font_sizes(elements_p1, line_boxes_p2, scale_factor)

    generate_html(image_path, elements_p1, font_sizes, line_boxes_p2, orig_w, orig_h, output_path)
    
    return str(output_path)


# ENTRY POINT
if __name__ == "__main__":
    DATA_DIR   = "task_b_layout_extraction/data"
    OUTPUT_DIR = "task_b_layout_extraction/output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Proses batch konversi seluruh dokumen 1 - 5
    for i in range(1, 6):
        fname = (
            f"Profile Image Studio  Let's Enable Digital Transformation "
            f"with Us (2)_page-000{i}.jpg"
        )
        img_path = Path(DATA_DIR) / fname
        
        if img_path.exists():
            process_slide(img_path, OUTPUT_DIR)