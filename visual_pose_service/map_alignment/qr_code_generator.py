import qrcode
from PIL import Image, ImageDraw
from fpdf import FPDF
import json
import os

# Parameters
ids = ["mobili_vlp_anchor_01", "mobili_vlp_anchor_02", "mobili_vlp_anchor_03"]
qr_error_correction = qrcode.constants.ERROR_CORRECT_H
dot_size = 32  # module pixel size
margin = 40  # outer margin in pixels
fill_color = (124, 83, 237)
bg_color = (255, 255, 255)
corner_radius = dot_size // 3  # rounded corner radius (~1/3 of module)


def draw_rounded_rect(draw, xy, radius, fill):
    x0, y0, x1, y1 = xy
    # Center rectangle
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    # Four rounded corners
    draw.pieslice([x0, y0, x0 + 2 * radius, y0 + 2 * radius], 180, 270, fill=fill)  # top-left
    draw.pieslice([x1 - 2 * radius, y0, x1, y0 + 2 * radius], 270, 360, fill=fill)  # top-right
    draw.pieslice([x0, y1 - 2 * radius, x0 + 2 * radius, y1], 90, 180, fill=fill)  # bottom-left
    draw.pieslice([x1 - 2 * radius, y1 - 2 * radius, x1, y1], 0, 90, fill=fill)  # bottom-right


def generate_qr_image(data):
    qr = qrcode.QRCode(
        version=None, error_correction=qr_error_correction, box_size=10, border=1  # auto version
    )
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    num_modules = len(matrix)
    img_size = dot_size * num_modules + 2 * margin
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw modules with rounded rects
    for y in range(num_modules):
        for x in range(num_modules):
            if matrix[y][x]:
                x0 = x * dot_size + margin
                y0 = y * dot_size + margin
                x1 = x0 + dot_size
                y1 = y0 + dot_size
                draw_rounded_rect(draw, (x0, y0, x1, y1), corner_radius, fill_color)

    # Optional centered logo
    logo_path = "artifacts/logo.png"
    try:
        logo = Image.open(logo_path).convert("RGBA")
        logo_size = dot_size * 7 + 2
        logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
        logo_pos = ((img_size - logo_size) // 2, (img_size - logo_size) // 2)
        img.paste(logo, logo_pos, logo)
    except FileNotFoundError:
        print("Warning: logo.png not found, skipping logo overlay")

    return img


# Target physical size (mm), e.g. 100 mm square
qr_mm_size = 200
dpi = 300
output_dir = "Data/anchors"
os.makedirs(output_dir, exist_ok=True)

# Generate QR images and PDF
pdf = FPDF(unit="mm", format="A4")
for anchor_id in ids:
    qr_img = generate_qr_image(anchor_id)

    # Resize for print resolution
    qr_px_size = int(qr_mm_size * dpi / 25.4)
    qr_img_resized = qr_img.resize((qr_px_size, qr_px_size), Image.NEAREST)

    # White A4 canvas, QR centered
    a4_width_px = int(210 * dpi / 25.4)
    a4_height_px = int(297 * dpi / 25.4)
    canvas = Image.new("RGB", (a4_width_px, a4_height_px), "white")
    offset = ((a4_width_px - qr_px_size) // 2, (a4_height_px - qr_px_size) // 2)
    canvas.paste(qr_img_resized, offset)

    png_path = os.path.join(output_dir, f"{anchor_id}.png")
    qr_img.save(png_path)

    temp_a4_path = os.path.join(output_dir, f"{anchor_id}_a4.jpg")
    canvas.save(temp_a4_path, "JPEG", quality=95)

    pdf.add_page()
    pdf.image(temp_a4_path, x=0, y=0, w=210, h=297)

pdf_path = os.path.join(output_dir, "anchors.pdf")
pdf.output(pdf_path)

anchors_json = {
    "qr_size_m": qr_mm_size / 1000.0,  # global QR size in meters
    "qr_ids": ids,
    "qr_distances_m": {}  # optional pairwise distances
}
json_path = os.path.join(output_dir, "qr.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(anchors_json, f, indent=2, ensure_ascii=False)

print(f"Done.\nPDF: {pdf_path}\nJSON: {json_path}")
