#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Распознавание подписей на советских генштабовских картах.

Скачивает тайлы генштабовки (nakarte.me) по кругу вокруг центра, распознаёт
подписи и ищет признаки исчезнувших мест: «(нежил.)», «ур. …», «разв.»,
«мельн.», «бывш.», «кладб.». Результат — файл ggc_points.json для вкладки
«Анализ» в poisk-karta.html и таблица ggc_points.csv.

Установка (один раз):
  1. Python 3.9+ : https://www.python.org/downloads/ (при установке отметь "Add to PATH")
  2. Tesseract OCR с русским языком:
       Windows: https://github.com/UB-Mannheim/tesseract/wiki — при установке
                в «Additional language data» отметь Russian
       macOS:   brew install tesseract tesseract-lang
       Linux:   sudo apt install tesseract-ocr tesseract-ocr-rus
  3. Библиотеки:  pip install requests pillow numpy pytesseract

Запуск (пример: 100 км вокруг Нижнего Новгорода, карта 1:100 000):
  python ggc_ocr.py --lat 56.3269 --lon 44.0059 --radius 100

Подробнее (1:50 000, мелкие деревни, но в ~4 раза дольше) — лучше на меньшем радиусе:
  python ggc_ocr.py --lat 56.10 --lon 43.50 --radius 30 --layer ggc500

Скачанные тайлы и распознанные куски кэшируются в папке ggc_cache —
прерванный запуск можно продолжить той же командой.
"""
import argparse, base64, concurrent.futures as cf, difflib, io, json, math, os, random, re, shutil, sys, time

try:
    import requests
    import numpy as np
    from PIL import Image, ImageOps
    import pytesseract
except ImportError as e:
    print("Не хватает библиотеки:", e.name)
    print("Установи: pip install requests pillow numpy pytesseract")
    sys.exit(1)

LAYERS = {
    # name: (zoom, описание)
    "ggc1000": (13, "Генштаб 1:100 000"),
    "ggc500":  (14, "Генштаб 1:50 000"),
    "ggc2000": (12, "Генштаб 1:200 000"),
}
DEFAULT_URL = "https://{s}.tiles.nakarte.me/{layer}/{z}/{x}/{y}"
UA = "poisk-karta-ocr/1.0 (personal use; slow polite download)"

# ---------------------------------------------------------------- tiles math
def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y

def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def haversine(a1, o1, a2, o2):
    r = math.radians
    d = math.sin(r(a2 - a1) / 2) ** 2 + math.cos(r(a1)) * math.cos(r(a2)) * math.sin(r(o2 - o1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(d))

def tiles_in_circle(lat, lon, rkm, z):
    dlat = rkm / 111.0
    dlon = rkm / (111.0 * math.cos(math.radians(lat)))
    x0, y0 = deg2tile(lat + dlat, lon - dlon, z)
    x1, y1 = deg2tile(lat - dlat, lon + dlon, z)
    out = set()
    for x in range(int(x0), int(x1) + 1):
        for y in range(int(y0), int(y1) + 1):
            la_n, lo_w = tile2deg(x, y, z)
            la_s, lo_e = tile2deg(x + 1, y + 1, z)
            ca = min(max(lat, la_s), la_n)
            co = min(max(lon, lo_w), lo_e)
            if haversine(lat, lon, ca, co) <= rkm * 1000:
                out.add((x, y))
    return out

# ---------------------------------------------------------------- download
class Downloader:
    def __init__(self, url, layer, z, tms, cache, delay):
        self.url, self.layer, self.z, self.tms, self.cache, self.delay = url, layer, z, tms, cache, delay
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Referer": "https://nakarte.me/"})

    def path(self, x, y):
        return os.path.join(self.cache, "tiles", self.layer, str(self.z), str(x), f"{y}.img")

    def get(self, x, y):
        p = self.path(x, y)
        if os.path.exists(p):
            return p if os.path.getsize(p) > 0 else None
        os.makedirs(os.path.dirname(p), exist_ok=True)
        yy = (2 ** self.z - 1 - y) if self.tms else y
        url = self.url.format(s=random.choice("abc"), layer=self.layer, z=self.z, x=x, y=yy)
        for attempt in range(5):
            try:
                r = self.s.get(url, timeout=30)
                if r.status_code == 404:
                    open(p, "wb").close()          # нет покрытия — запоминаем пустым
                    return None
                if r.status_code == 200 and r.content:
                    with open(p + ".part", "wb") as f:
                        f.write(r.content)
                    os.replace(p + ".part", p)
                    time.sleep(self.delay)
                    return p
                time.sleep(2 + attempt * 3)
            except requests.RequestException:
                time.sleep(2 + attempt * 3)
        return None

# ---------------------------------------------------------------- OCR
LAT2CYR = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
                         "k": "к", "m": "м", "h": "н", "b": "в", "t": "т", "u": "и", "n": "п", "r": "г",
                         "3": "з", "0": "о", "6": "б"})

def norm(w):
    w = w.lower().translate(LAT2CYR)
    return re.sub(r"[^а-яё\.]", "", w)

KEYS = [
    # (вид, образец, мин. похожесть, описание)
    ("g_vanished", "нежил", 0.78, "нежил."),
    ("g_vanished", "бывш",  0.85, "бывш."),
    ("g_ruins",    "разв",  0.85, "разв."),
    ("g_ruins",    "развалины", 0.8, "развалины"),
    ("g_mill",     "мельн", 0.8, "мельн."),
    ("g_mill",     "мельница", 0.8, "мельница"),
    ("g_cem",      "кладб", 0.8, "кладб."),
]

def match_key(raw):
    w = norm(raw).strip(".")
    if not w:
        return None
    if w == "ур" and "." in raw:
        return ("g_ur", "ур.")
    if len(w) < 4:
        return None
    best = None
    for kind, pat, thr, lbl in KEYS:
        # окно по слову: OCR часто приклеивает скобку как «г», «з», «{»
        r = max(difflib.SequenceMatcher(None, w[i:i + len(pat)], pat).ratio()
                for i in range(0, max(1, min(3, len(w) - len(pat) + 2))))
        if r >= thr and (best is None or r > best[0]):
            best = (r, kind, lbl)
    return (best[1], best[2]) if best else None

def text_mask(img, scale=2):
    """Оставляем только тёмные малонасыщенные пиксели (чёрные подписи)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    mx, mn = a.max(axis=2), a.min(axis=2)
    gray = np.where((mx - mn) < 75, mx, 255).astype(np.uint8)
    g = Image.fromarray(gray)
    if scale != 1:
        g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
    return g.point(lambda v: 0 if v < 135 else 255)

def ocr_words(img, scale=2):
    m = text_mask(img, scale)
    d = pytesseract.image_to_data(m, lang="rus", config="--oem 1 --psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except ValueError:
            conf = -1
        words.append({
            "t": t, "conf": conf,
            "x": d["left"][i] / scale, "y": d["top"][i] / scale,
            "w": d["width"][i] / scale, "h": d["height"][i] / scale,
            "line": (d["block_num"][i], d["par_num"][i], d["line_num"][i]), "n": d["word_num"][i],
        })
    return words

def line_text(words, line):
    ws = sorted([w for w in words if w["line"] == line], key=lambda w: w["n"])
    return " ".join(w["t"] for w in ws)

def find_labels(words):
    found = []
    for w in words:
        k = match_key(w["t"])
        if not k:
            continue
        kind, lbl = k
        name, anchor = "", w
        if kind == "g_ur":
            nxt = sorted([v for v in words if v["line"] == w["line"] and v["n"] > w["n"]], key=lambda v: v["n"])
            if not nxt or not re.match(r"^[А-ЯЁA-Z]", nxt[0]["t"]):
                continue
            name = " ".join(v["t"] for v in nxt[:2])
        else:
            # название обычно стоит над пометкой «(нежил.)», иногда перед ней в той же строке
            cx = w["x"] + w["w"] / 2
            above = [v for v in words if v is not w and v["y"] + v["h"] <= w["y"] + 2
                     and w["y"] - (v["y"] + v["h"]) < 2.2 * max(w["h"], 8)
                     and abs((v["x"] + v["w"] / 2) - cx) < max(w["w"], v["w"]) * 0.9
                     and re.match(r"^[А-ЯЁA-Z]", v["t"])]
            same = [v for v in words if v["line"] == w["line"] and v["n"] < w["n"] and re.match(r"^[А-ЯЁA-Z]", v["t"])]
            if above:
                v = min(above, key=lambda v: w["y"] - v["y"])
                name, anchor = line_text(words, v["line"]), v
            elif same:
                name = " ".join(v["t"] for v in sorted(same, key=lambda v: v["n"]))
        found.append({"kind": kind, "label": lbl, "name": re.sub(r"\s+", " ", name).strip(" .,()"),
                      "text": w["t"], "conf": w["conf"],
                      "bx": min(w["x"], anchor["x"]), "by": min(w["y"], anchor["y"]),
                      "bx2": max(w["x"] + w["w"], anchor["x"] + anchor["w"]),
                      "by2": max(w["y"] + w["h"], anchor["y"] + anchor["h"])})
    return found

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Поиск «нежил.», «ур.», «разв.», «мельн.» на генштабовках")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius", type=float, default=30, help="радиус, км (по умолчанию 30)")
    ap.add_argument("--layer", default="ggc1000", help="ggc1000 (1:100 000, по умолчанию), ggc500 (1:50 000), ggc2000")
    ap.add_argument("--zoom", type=int, default=None, help="уровень тайлов (по умолчанию родной для слоя)")
    ap.add_argument("--url", default=DEFAULT_URL, help="шаблон адреса тайлов: {s},{layer},{z},{x},{y}")
    ap.add_argument("--no-tms", action="store_true", help="ось Y не перевёрнута (для своих источников)")
    ap.add_argument("--out", default="ggc_result", help="папка для результата")
    ap.add_argument("--cache", default="ggc_cache", help="папка кэша")
    ap.add_argument("--threads", type=int, default=4, help="потоков скачивания (не больше 4, сервер бесплатный)")
    ap.add_argument("--delay", type=float, default=0.05, help="пауза после каждого тайла, сек")
    ap.add_argument("--block", type=int, default=8, help="размер куска для распознавания, тайлов")
    ap.add_argument("--tesseract", default=None, help="путь к tesseract, если не находится сам")
    a = ap.parse_args()

    # tesseract
    if a.tesseract:
        pytesseract.pytesseract.tesseract_cmd = a.tesseract
    elif not shutil.which("tesseract"):
        for p in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                  "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    try:
        langs = pytesseract.get_languages(config="")
    except Exception as e:
        print("Не найден Tesseract OCR. Установи его (см. начало файла) или укажи путь: --tesseract \"C:\\...\\tesseract.exe\"")
        sys.exit(1)
    if "rus" not in langs:
        print("В Tesseract нет русского языка (rus). Переустанови с отмеченным Russian или положи rus.traineddata в папку tessdata.")
        sys.exit(1)

    z = a.zoom or LAYERS.get(a.layer, (13, ""))[0]
    tiles = tiles_in_circle(a.lat, a.lon, a.radius, z)
    print(f"Слой {a.layer}, зум {z}: {len(tiles)} тайлов в круге {a.radius} км.")
    dl = Downloader(a.url, a.layer, z, not a.no_tms, a.cache, a.delay)

    # 1. скачивание
    t0, done, have = time.time(), 0, {}
    with cf.ThreadPoolExecutor(max_workers=max(1, min(a.threads, 6))) as ex:
        futs = {ex.submit(dl.get, x, y): (x, y) for (x, y) in tiles}
        for f in cf.as_completed(futs):
            done += 1
            p = f.result()
            if p:
                have[futs[f]] = p
            if done % 50 == 0 or done == len(tiles):
                el = time.time() - t0
                eta = el / done * (len(tiles) - done)
                print(f"\r  скачано {done}/{len(tiles)}, с картой {len(have)}, осталось ~{eta/60:.0f} мин   ", end="", flush=True)
    print()
    if not have:
        print("Ни одного тайла не скачалось: нет интернета или сервер недоступен.")
        sys.exit(1)

    # 2. распознавание кусками с перекрытием в 1 тайл
    B = max(2, a.block)
    xs = [t[0] for t in have]; ys = [t[1] for t in have]
    blocks = []
    for bx in range(min(xs), max(xs) + 1, B - 1):
        for by in range(min(ys), max(ys) + 1, B - 1):
            if any((x, y) in have for x in range(bx, bx + B) for y in range(by, by + B)):
                blocks.append((bx, by))
    ocr_dir = os.path.join(a.cache, "ocr", f"{a.layer}_{z}_{B}")
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    def do_block(b):
        bx, by = b
        cp = os.path.join(ocr_dir, f"{bx}_{by}.json")
        if os.path.exists(cp):
            with open(cp, encoding="utf-8") as f:
                return json.load(f)
        mos = Image.new("RGB", (B * 256, B * 256), (255, 255, 255))
        for i in range(B):
            for j in range(B):
                p = have.get((bx + i, by + j))
                if p:
                    try:
                        mos.paste(Image.open(p).convert("RGB"), (i * 256, j * 256))
                    except Exception:
                        pass
        labels = find_labels(ocr_words(mos))
        res = []
        for L in labels:
            cx, cy = (L["bx"] + L["bx2"]) / 2, (L["by"] + L["by2"]) / 2
            lat, lon = tile2deg(bx + cx / 256, by + cy / 256, z)
            pad = 40
            box = (max(0, int(L["bx"] - pad)), max(0, int(L["by"] - pad)),
                   min(mos.width, int(L["bx2"] + pad)), min(mos.height, int(L["by2"] + pad)))
            crop = mos.crop(box)
            crop.thumbnail((260, 160))
            buf = io.BytesIO(); crop.save(buf, "JPEG", quality=72)
            res.append({"lat": round(lat, 6), "lon": round(lon, 6), "kind": L["kind"], "label": L["label"],
                        "name": L["name"], "text": L["text"], "conf": round(L["conf"], 1),
                        "crop": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        return res

    print(f"Распознаю {len(blocks)} кусков (по {B}×{B} тайлов)…")
    t0, done, raw = time.time(), 0, []
    workers = max(1, (os.cpu_count() or 2) - 1)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(do_block, blocks):
            raw.extend(res); done += 1
            el = time.time() - t0
            print(f"\r  {done}/{len(blocks)}, найдено пометок: {len(raw)}, осталось ~{el/done*(len(blocks)-done)/60:.0f} мин   ", end="", flush=True)
    print()

    # 3. склейка дублей из перекрытий и фильтр по кругу
    raw = [p for p in raw if haversine(a.lat, a.lon, p["lat"], p["lon"]) <= a.radius * 1000]
    raw.sort(key=lambda p: (-len(p["name"]), -p["conf"]))
    pts = []
    for p in raw:
        if any(q["kind"] == p["kind"] and haversine(p["lat"], p["lon"], q["lat"], q["lon"]) < 250 for q in pts):
            continue
        pts.append(p)
    for i, p in enumerate(pts):
        p["id"] = f"g{z}_{i}"

    out = {"format": "poisk-ggc-ocr", "version": 1, "layer": a.layer, "zoom": z,
           "params": {"lat": a.lat, "lon": a.lon, "radius": a.radius}, "created": time.strftime("%Y-%m-%d %H:%M"),
           "points": pts}
    jp = os.path.join(a.out, "ggc_points.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    with open(os.path.join(a.out, "ggc_points.csv"), "w", encoding="utf-8-sig") as f:
        f.write("вид;пометка;название;широта;долгота;уверенность\n")
        for p in pts:
            f.write(f'{p["kind"]};{p["label"]};"{p["name"]}";{p["lat"]};{p["lon"]};{p["conf"]}\n')
    by = {}
    for p in pts:
        by[p["label"]] = by.get(p["label"], 0) + 1
    print("Готово:", ", ".join(f"{k}: {v}" for k, v in sorted(by.items(), key=lambda x: -x[1])) or "ничего не найдено")
    print("Файл для карты:", os.path.abspath(jp))
    print("Открой poisk-karta.html → вкладка «Анализ» → «Загрузить результаты распознавания».")

if __name__ == "__main__":
    main()
