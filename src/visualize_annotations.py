
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

patient_id = "00436515-870c-4b36-a041-de91049b9ab4"

image_path = (
    ROOT / "data" / "processed" /
    "images_224" / f"{patient_id}.png"
)

labels_path = ROOT / "data" / "raw" / "stage_2_train_labels.csv"

df = pd.read_csv(labels_path)

boxes = df[
    (df["patientId"] == patient_id) &
    (df["Target"] == 1)
]

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# RSNA annotation coordinates refer to original 1024x1024 images.
scale_x = image.width / 1024
scale_y = image.height / 1024

for _, row in boxes.iterrows():
    x1 = row["x"] * scale_x
    y1 = row["y"] * scale_y
    x2 = (row["x"] + row["width"]) * scale_x
    y2 = (row["y"] + row["height"]) * scale_y

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3,
    )

output = ROOT / "reports" / "annotation_example.png"
output.parent.mkdir(parents=True, exist_ok=True)

image.save(output)

print("SUCCESS!")
print("Bounding boxes drawn:", len(boxes))
print("Saved to:", output)
