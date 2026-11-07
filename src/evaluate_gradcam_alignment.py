
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/processed/images_224"
LABELS_PATH = ROOT / "data/raw/stage_2_train_labels.csv"
MODEL_PATH = ROOT / "checkpoints/resnet18_stage2_best.pth"
OUTPUT_PATH = ROOT / "reports/gradcam_alignment_20.csv"

SAMPLE_SIZE = 20
SEED = 42

labels = pd.read_csv(LABELS_PATH)
opacity = labels[labels["Target"] == 1].copy()

# Use one row per image for sampling.
available_ids = sorted(
    set(opacity["patientId"]) &
    {path.stem for path in IMAGE_DIR.glob("*.png")}
)

# Exclude held-out test images to avoid using the test set
# for exploratory analysis.
test_manifest = ROOT / "data/processed/test_split.csv"
if test_manifest.exists():
    test_df = pd.read_csv(test_manifest)
    id_column = next(
        (col for col in ["patientId", "patient_id", "image_id"]
         if col in test_df.columns),
        None,
    )
    if id_column:
        test_ids = set(
            test_df[id_column].astype(str).str.replace(
                r"\.png$", "", regex=True
            )
        )
        available_ids = [
            image_id for image_id in available_ids
            if image_id not in test_ids
        ]
    else:
        raise ValueError(
            "Could not identify image ID column in test_split.csv"
        )
else:
    raise FileNotFoundError(
        "test_split.csv not found. Verify split manifest first."
    )

rng = np.random.default_rng(SEED)
sample_ids = rng.choice(
    available_ids,
    size=min(SAMPLE_SIZE, len(available_ids)),
    replace=False,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_to_idx = checkpoint["class_to_idx"]
opacity_index = class_to_idx["Lung Opacity"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.485, 0.485],
        std=[0.229, 0.229, 0.229],
    ),
])

results = []

for number, image_id in enumerate(sample_ids, start=1):
    image = Image.open(
        IMAGE_DIR / f"{image_id}.png"
    ).convert("RGB")

    tensor = preprocess(image).unsqueeze(0)

    captured = {}

    def forward_hook(module, inputs, output):
        captured["activations"] = output
        output.register_hook(
            lambda grad: captured.update(gradients=grad)
        )

    handle = model.layer4[-1].register_forward_hook(
        forward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            scores = model(tensor)
            predicted_index = int(scores.argmax(dim=1).item())

            # Explain Lung Opacity score for every image.
            scores[0, opacity_index].backward()

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * activations).sum(dim=1)
        )[0].cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        cam_image = Image.fromarray(
            np.uint8(cam * 255)
        ).resize(
            (224, 224),
            Image.Resampling.BILINEAR,
        )

        cam = np.asarray(
            cam_image, dtype=np.float32
        ) / 255.0

    finally:
        handle.remove()

    mask = np.zeros((224, 224), dtype=bool)

    image_boxes = opacity[
        opacity["patientId"] == image_id
    ]

    for _, box in image_boxes.iterrows():
        x1 = int(np.floor(box["x"] * 224 / 1024))
        y1 = int(np.floor(box["y"] * 224 / 1024))
        x2 = int(np.ceil(
            (box["x"] + box["width"]) * 224 / 1024
        ))
        y2 = int(np.ceil(
            (box["y"] + box["height"]) * 224 / 1024
        ))

        x1, x2 = np.clip([x1, x2], 0, 224)
        y1, y2 = np.clip([y1, y2], 0, 224)

        mask[y1:y2, x1:x2] = True

    total_energy = float(cam.sum())

    energy_inside = (
        float(cam[mask].sum() / total_energy)
        if total_energy > 1e-8
        else np.nan
    )

    results.append({
        "image_id": image_id,
        "predicted_lung_opacity": predicted_index == opacity_index,
        "annotation_boxes": len(image_boxes),
        "box_area_fraction": float(mask.mean()),
        "gradcam_energy_inside_boxes": energy_inside,
    })

    print(f"Processed {number}/{len(sample_ids)}")

df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)
df.to_csv(OUTPUT_PATH, index=False)

print("\nEXPERIMENT COMPLETE")
print("Images evaluated:", len(df))
print(
    "Mean Grad-CAM energy inside boxes:",
    round(
        df["gradcam_energy_inside_boxes"].mean() * 100,
        2,
    ),
    "%",
)
print(
    "Mean annotation area:",
    round(df["box_area_fraction"].mean() * 100, 2),
    "%",
)
print("Saved to:", OUTPUT_PATH)
