#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нарезает старые карты из папки maps-src/ на тайлы для сайта.

Каждая подпапка maps-src/<имя>/ — один слой (можно много листов).
Для каждого листа нужна картинка (jpg/png/tif) и рядом файл привязки
с тем же именем: .map (OziExplorer) или .kml (GroundOverlay).
GeoTIFF с привязкой можно класть без отдельного файла.
Необязательный maps-src/<имя>/layer.txt:
    title=Менде, 1850
    year=1850
    maxzoom=15
Результат: tiles/<имя>/{z}/{x}/{y}.webp и список слоёв maps.json.
Запускается GitHub Action'ом, вручную не нужно.
"""
import hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, time
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "maps-src")
OUT = os.path.join(ROOT, "tiles")
INDEX = os.path.join(ROOT, "maps.json")
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp", ".webp")
DATUMS = {"wgs 84": "EPSG:4326", "wgs84": "EPSG:4326", "pulkovo 1942": "EPSG:4284", "pulkovo 1942 (1)": "EPSG:4284",
          "pulkovo 1942 (2)": "EPSG:4284", "s-42": "EPSG:4284", "sk-42": "EPSG:4284"}

TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              "a b v g d e e zh z i y k l m n o p r s t u f h ts ch sh sch _ y _ e yu ya".split()))
def slug(s):
    s = "".join(TR.get(c, c) for c in s.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "map"

def run(cmd):
    print("$", " ".join(cmd[:6]), "…" if len(cmd) > 6 else "", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise RuntimeError("команда не выполнилась: " + cmd[0])
    return r.stdout

def img_size(path):
    info = json.loads(run(["gdalinfo", "-json", path]))
    return info["size"][0], info["size"][1], info

def parse_ozi(path, w, h):
    txt = open(path, "r", encoding="cp1251", errors="replace").read()
    lines = txt.splitlines()
    datum = lines[4].split(",")[0].strip().lower() if len(lines) > 4 else "wgs 84"
    srs = DATUMS.get(datum, "EPSG:4326")
    xy, ll = {}, {}
    for l in lines:
        m = re.match(r"^MMPXY\s*,\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", l)
        if m: xy[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r"^MMPLL\s*,\s*(\d+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", l)
        if m: ll[m.group(1)] = (float(m.group(2)), float(m.group(3)))   # lon, lat
    gcps = []
    for l in lines:                                   # калибровочные точки точнее углов
        if not re.match(r"^Point\d+", l): continue
        f = [s.strip() for s in l.split(",")]
        try:
            x, y = float(f[2]), float(f[3])
            lat = float(f[6]) + float(f[7] or 0) / 60
            lon = float(f[9]) + float(f[10] or 0) / 60
        except (ValueError, IndexError):
            continue
        if f[8] == "S": lat = -lat
        if f[11] == "W": lon = -lon
        gcps.append((x, y, lon, lat))
    # MMPLL всегда в WGS84
    if len(gcps) < 3:
        gcps = [(xy[k][0], xy[k][1], ll[k][0], ll[k][1]) for k in xy if k in ll]
        srs = "EPSG:4326"
    return gcps, srs

def parse_kml(path, w, h):
    t = ET.parse(path).getroot()
    def find(tag):
        for e in t.iter():
            if e.tag.split("}")[-1] == tag: return e
    q = find("LatLonQuad")
    if q is not None:
        c = [list(map(float, s.split(",")[:2])) for s in find_in(q, "coordinates").text.split()]
        ll_, lr, ur, ul = c[:4]
        return [(0, 0, ul[0], ul[1]), (w, 0, ur[0], ur[1]), (w, h, lr[0], lr[1]), (0, h, ll_[0], ll_[1])], "EPSG:4326"
    b = {k: float(find(k).text) for k in ("north", "south", "east", "west")}
    return [(0, 0, b["west"], b["north"]), (w, 0, b["east"], b["north"]), (w, h, b["east"], b["south"]), (0, h, b["west"], b["south"])], "EPSG:4326"
def find_in(node, tag):
    for e in node.iter():
        if e.tag.split("}")[-1] == tag: return e

def file_hash(paths):
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode()); h.update(str(os.path.getsize(p)).encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                h.update(b)
    return h.hexdigest()[:16]

def read_meta(folder):
    meta = {}
    p = os.path.join(folder, "layer.txt")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8", errors="replace"):
            if "=" in l:
                k, v = l.split("=", 1); meta[k.strip().lower()] = v.strip()
    return meta

def layers():
    if not os.path.isdir(SRC): return []
    out = []
    for name in sorted(os.listdir(SRC)):
        p = os.path.join(SRC, name)
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in sorted(os.listdir(p)) if not f.startswith(".")]
            imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
            if imgs: out.append((name, p, imgs, files))
        elif name.lower().endswith(IMG_EXT):
            base = os.path.splitext(p)[0]
            side = [base + e for e in (".map", ".MAP", ".kml", ".KML") if os.path.exists(base + e)]
            out.append((os.path.splitext(name)[0], SRC, [p], [p] + side))
    return out

def build_layer(name, folder, imgs, files, tmp):
    warped = []
    for img in imgs:
        base = os.path.splitext(img)[0]
        w, h, info = img_size(img)
        side = next((base + e for e in (".map", ".MAP", ".kml", ".KML") if os.path.exists(base + e)), None)
        if side:
            gcps, srs = parse_ozi(side, w, h) if side.lower().endswith(".map") else parse_kml(side, w, h)
            if len(gcps) < 3:
                print("  пропуск (мало точек привязки):", os.path.basename(img)); continue
            vrt = os.path.join(tmp, slug(os.path.basename(base)) + ".vrt")
            cmd = ["gdal_translate", "-of", "VRT", "-a_srs", srs]
            for x, y, lon, lat in gcps: cmd += ["-gcp", str(x), str(y), str(lon), str(lat)]
            run(cmd + [img, vrt])
            order = ["-tps"] if len(gcps) >= 6 else ["-order", "1"]
            src = vrt
        elif info.get("coordinateSystem", {}).get("wkt"):
            src, order = img, []
        else:
            print("  пропуск (нет файла привязки .map/.kml):", os.path.basename(img)); continue
        out = os.path.join(tmp, slug(os.path.basename(base)) + "_3857.tif")
        run(["gdalwarp", "-overwrite", "-t_srs", "EPSG:3857", "-r", "bilinear", "-dstalpha", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
             "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=IF_SAFER"] + order + [src, out])
        warped.append(out)
    if not warped: return None
    mosaic = os.path.join(tmp, "mosaic.vrt")
    run(["gdalbuildvrt", "-overwrite", mosaic] + warped)
    info = json.loads(run(["gdalinfo", "-json", mosaic]))
    res = abs(info["geoTransform"][1])
    meta = read_meta(folder) if folder != SRC else {}
    zmax = int(meta.get("maxzoom") or max(10, min(16, round(math.log2(156543.034 / res)))))
    zmin = int(meta.get("minzoom") or 8)
    dest = os.path.join(OUT, slug(name))
    shutil.rmtree(dest, ignore_errors=True)
    ext = "webp"
    try:
        run(["gdal2tiles.py", "--xyz", "-z", f"{zmin}-{zmax}", "-w", "none", "-r", "average", f"--processes={os.cpu_count() or 2}",
             "--tiledriver=WEBP", "--webp-quality=72", mosaic, dest])
    except RuntimeError:
        shutil.rmtree(dest, ignore_errors=True); ext = "png"
        run(["gdal2tiles.py", "--xyz", "-z", f"{zmin}-{zmax}", "-w", "none", "-r", "average", f"--processes={os.cpu_count() or 2}", mosaic, dest])
    for junk in ("tilemapresource.xml", "openlayers.html", "leaflet.html", "googlemaps.html"):
        try: os.remove(os.path.join(dest, junk))
        except FileNotFoundError: pass
    # границы в градусах
    corners = info["cornerCoordinates"]
    def to_ll(x, y):
        return (math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2), math.degrees(x / 6378137.0))
    s, w_ = to_ll(corners["lowerLeft"][0], corners["lowerLeft"][1])
    n, e = to_ll(corners["upperRight"][0], corners["upperRight"][1])
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(dest) for f in fs)
    year = meta.get("year") or (re.search(r"(1[6-9]\d\d)", name).group(1) if re.search(r"(1[6-9]\d\d)", name) else "")
    return {"id": slug(name), "title": meta.get("title") or name.replace("_", " ").replace("-", " "), "year": year,
            "url": f"tiles/{slug(name)}/{{z}}/{{x}}/{{y}}.{ext}", "minzoom": zmin, "maxzoom": zmax,
            "bounds": [[round(s, 5), round(w_, 5)], [round(n, 5), round(e, 5)]], "sheets": len(warped),
            "sizeMB": round(size / 1e6, 1), "built": time.strftime("%Y-%m-%d %H:%M")}

def main():
    old = {}
    if os.path.exists(INDEX):
        try: old = {m["id"]: m for m in json.load(open(INDEX, encoding="utf-8")).get("maps", [])}
        except Exception: old = {}
    result, seen = [], set()
    for name, folder, imgs, files in layers():
        sid = slug(name); seen.add(sid)
        hsh = file_hash(files + ([os.path.join(folder, "layer.txt")] if os.path.exists(os.path.join(folder, "layer.txt")) and folder != SRC else []))
        if sid in old and old[sid].get("hash") == hsh and os.path.isdir(os.path.join(OUT, sid)):
            print("без изменений:", name); result.append(old[sid]); continue
        print("== слой:", name, f"({len(imgs)} лист.)", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                m = build_layer(name, folder, imgs, files, tmp)
            except Exception as ex:
                print("  ОШИБКА:", ex); m = None
        if m:
            m["hash"] = hsh; result.append(m); print("  готово:", m["title"], m["sizeMB"], "МБ, зум", m["minzoom"], "-", m["maxzoom"])
    for sid in set(old) - seen:
        print("удалён слой:", sid); shutil.rmtree(os.path.join(OUT, sid), ignore_errors=True)
    total = sum(m.get("sizeMB", 0) for m in result)
    json.dump({"updated": time.strftime("%Y-%m-%d %H:%M"), "totalMB": round(total, 1), "maps": result},
              open(INDEX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Всего слоёв: {len(result)}, тайлов на {total:.0f} МБ" + ("  ВНИМАНИЕ: сайт GitHub Pages ограничен ~1 ГБ" if total > 800 else ""))

if __name__ == "__main__":
    main()
