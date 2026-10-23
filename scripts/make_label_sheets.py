"""Blind label sheets: tiles of detector boxes, no model output."""
import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "reports", "real-baseline-shipped", "results.json")
PHOTOS = os.path.join(ROOT, "data", "fb-eval-pack", "user-saved")
OUT = os.path.join(ROOT, "data", "fb-eval-pack", "label-sheets")
PAD, LONG, COLS, ROWS = 0.08, 360, 4, 3
CELL_W, CELL_H = 400, 500
BG = (60, 60, 60)
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp")


def font(size):
    for p in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def tile(img, box):
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    c = img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                  min(img.width, int(x2 + pw + .5)), min(img.height, int(y2 + ph + .5))))
    scale = max(1.0, LONG / max(c.size))
    return c.resize((round(c.width * scale), round(c.height * scale)), Image.LANCZOS)


def main():
    boxes = json.load(open(RESULTS))["boxes"]
    files = sorted(f for f in os.listdir(PHOTOS) if f.lower().endswith(PHOTO_EXT))
    os.makedirs(OUT, exist_ok=True)
    ft = font(22)
    index = {"photos": {}, "sheets": []}
    per = COLS * ROWS
    for n, f in enumerate(files, 1):
        index["photos"][str(n)] = f
        img = Image.open(os.path.join(PHOTOS, f)).convert("RGB")
        rows = [r for r in boxes if r["photo"] == f]
        for k, s in enumerate(range(0, len(rows), per), 1):
            sheet = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), BG)
            dr = ImageDraw.Draw(sheet)
            ids = []
            for i, r in enumerate(rows[s:s + per]):
                bid = s + i + 1
                t = tile(img, r["box"])
                cx, cy = (i % COLS) * CELL_W, (i // COLS) * CELL_H
                sheet.paste(t, (cx + (CELL_W - t.width) // 2, cy + 8))
                dr.text((cx + 12, cy + 8 + LONG + 14), f"p{n}-#{bid}  h={r['height']}", fill=(255, 255, 255), font=ft)
                ids.append({"photo": n, "box_id": bid, "box": r["box"]})
            name = f"photo{n}-sheet{k}.jpg"
            sheet.save(os.path.join(OUT, name), "JPEG", quality=90)
            index["sheets"].append({"file": name, "boxes": ids})
    index["total_boxes"] = sum(len(s["boxes"]) for s in index["sheets"])
    index["total_sheets"] = len(index["sheets"])
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(index["total_boxes"], index["total_sheets"], len(boxes))


main()
