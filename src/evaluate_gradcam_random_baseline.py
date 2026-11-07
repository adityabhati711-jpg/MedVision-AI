
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# MEDVISION AI
# Grad-CAM vs Random Baseline
# Validation-only experiment
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/processed/images_224"
LABELS_PATH = ROOT / "data/raw/stage_2_train_labels.csv"

TRAIN_PATH = ROOT / "data/processed/train_split.csv"
VALIDATION_PATH = ROOT / "data/processed/validation_split.csv"
TEST_PATH = ROOT / "data/processed/test_split.csv"

MODEL_PATH = ROOT / "checkpoints/resnet18_stage2_best.pth"
REPORT_DIR = ROOT / "reports"

SAMPLE_SIZE = 100
RANDOM_TRIALS = 100
SEED = 42

IMAGE_SIZE = 224
ORIGINAL_SIZE = 1024

rng = np.random.default_rng(SEED)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("Loading dataset information...")

labels = pd.read_csv(LABELS_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

if "patientId" not in validation_df.columns:
    raise ValueError(
        "patientId column missing from validation_split.csv"
    )

if "class" not in validation_df.columns:
    raise ValueError(
        "class column missing from validation_split.csv"
    )

if "patientId" not in test_df.columns:
    raise ValueError(
        "patientId column missing from test_split.csv"
    )

if not {"patientId", "Target", "x", "y", "width", "height"}.issubset(
    labels.columns
):
    raise ValueError(
        "Required annotation columns missing from labels CSV"
    )

labels["patientId"] = labels["patientId"].astype(str)

validation_df["patientId"] = (
    validation_df["patientId"].astype(str)
)

test_df["patientId"] = test_df["patientId"].astype(str)

opacity = labels[labels["Target"] == 1].copy()

validation_ids = set(
    validation_df.loc[
        validation_df["class"] == "Lung Opacity",
        "patientId"
    ]
)

test_ids = set(test_df["patientId"])

available_images = {
    path.stem for path in IMAGE_DIR.glob("*.png")
}

eligible_ids = sorted(
    (
        set(opacity["patientId"])
        & validation_ids
        & available_images
    ) - test_ids
)

print("Eligible validation images:", len(eligible_ids))

if len(eligible_ids) < SAMPLE_SIZE:
    raise ValueError(
        f"Only {len(eligible_ids)} eligible validation images "
        f"available; requested {SAMPLE_SIZE}."
    )

sample_ids = rng.choice(
    eligible_ids,
    size=SAMPLE_SIZE,
    replace=False,
)

print("Selected images:", len(sample_ids))
print("Random seed:", SEED)


# ============================================================
# STEP 2: LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained ResNet18 model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

model = resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    3,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

opacity_index = checkpoint["class_to_idx"]["Lung Opacity"]

print("Model loaded successfully.")
print("Lung Opacity class index:", opacity_index)


# ============================================================
# STEP 3: IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485] * 3,
        std=[0.229] * 3,
    ),
])


# ============================================================
# STEP 4: GRAD-CAM
# ============================================================

captured = {}


def forward_hook(module, inputs, output):
    captured["activation"] = output

    output.register_hook(
        lambda gradient: captured.update(
            gradient=gradient
        )
    )


handle = model.layer4[-1].register_forward_hook(
    forward_hook
)


def generate_cam(image):
    tensor = preprocess(image).unsqueeze(0)

    captured.clear()
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        scores = model(tensor)

        predicted = int(
            scores.argmax(dim=1).item()
        )

        # Explain the Lung Opacity class score.
        scores[0, opacity_index].backward()

    activation = captured["activation"].detach()
    gradient = captured["gradient"].detach()

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = torch.relu(
        (weights * activation).sum(dim=1)
    )[0].cpu().numpy()

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    cam = np.asarray(
        cam,
        dtype=np.float32,
    ) / 255.0

    return cam, predicted


# ============================================================
# STEP 5: RADIOLOGIST ANNOTATION BOXES
# ============================================================

def get_boxes(image_id):
    rows = opacity[
        opacity["patientId"] == image_id
    ]

    boxes = []

    for _, row in rows.iterrows():
        x = int(
            np.floor(
                row["x"] * IMAGE_SIZE / ORIGINAL_SIZE
            )
        )

        y = int(
            np.floor(
                row["y"] * IMAGE_SIZE / ORIGINAL_SIZE
            )
        )

        w = max(
            1,
            int(
                np.ceil(
                    row["width"]
                    * IMAGE_SIZE
                    / ORIGINAL_SIZE
                )
            ),
        )

        h = max(
            1,
            int(
                np.ceil(
                    row["height"]
                    * IMAGE_SIZE
                    / ORIGINAL_SIZE
                )
            ),
        )

        w = min(w, IMAGE_SIZE)
        h = min(h, IMAGE_SIZE)

        x = int(
            np.clip(
                x,
                0,
                IMAGE_SIZE - w,
            )
        )

        y = int(
            np.clip(
                y,
                0,
                IMAGE_SIZE - h,
            )
        )

        boxes.append((x, y, w, h))

    return boxes


def mask_from_boxes(boxes):
    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=bool,
    )

    for x, y, w, h in boxes:
        mask[y:y + h, x:x + w] = True

    return mask


# ============================================================
# STEP 6: RANDOM BASELINE
# ============================================================

def randomize_boxes(boxes):
    randomized = []

    for _, _, w, h in boxes:
        x = int(
            rng.integers(
                0,
                IMAGE_SIZE - w + 1,
            )
        )

        y = int(
            rng.integers(
                0,
                IMAGE_SIZE - h + 1,
            )
        )

        randomized.append((x, y, w, h))

    return randomized


# ============================================================
# STEP 7: RUN EXPERIMENT
# ============================================================

results = []

print("\nStarting validation-only experiment...")

try:
    for index, image_id in enumerate(
        sample_ids,
        start=1,
    ):
        image_path = IMAGE_DIR / f"{image_id}.png"

        with Image.open(image_path) as loaded_image:
            image = loaded_image.convert("RGB")

        cam, predicted = generate_cam(image)

        boxes = get_boxes(image_id)

        true_mask = mask_from_boxes(boxes)

        total_energy = float(cam.sum())

        if total_energy <= 1e-8:
            true_energy = np.nan
            random_mean = np.nan

        else:
            true_energy = float(
                cam[true_mask].sum() / total_energy
            )

            random_scores = []

            for _ in range(RANDOM_TRIALS):
                random_boxes = randomize_boxes(boxes)

                random_mask = mask_from_boxes(
                    random_boxes
                )

                random_energy = float(
                    cam[random_mask].sum()
                    / total_energy
                )

                random_scores.append(
                    random_energy
                )

            random_mean = float(
                np.mean(random_scores)
            )

        results.append({
            "image_id": image_id,
            "dataset_split": "validation",
            "predicted_lung_opacity": (
                predicted == opacity_index
            ),
            "annotation_boxes": len(boxes),
            "annotation_area_fraction": float(
                true_mask.mean()
            ),
            "true_box_energy": true_energy,
            "random_box_energy": random_mean,
            "energy_difference": (
                true_energy - random_mean
            ),
        })

        if index % 10 == 0:
            print(
                f"Processed {index}/{SAMPLE_SIZE}"
            )

finally:
    handle.remove()


# ============================================================
# STEP 8: SAVE RESULTS
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.DataFrame(results)

csv_path = (
    REPORT_DIR
    / "gradcam_random_baseline_validation_100.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

valid = df.dropna(
    subset=[
        "true_box_energy",
        "random_box_energy",
    ]
)


# ============================================================
# STEP 9: PRINT SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("VALIDATION-ONLY EXPERIMENT COMPLETE")
print("=" * 55)

print("Images evaluated:", len(df))

print(
    "Correct Lung Opacity predictions:",
    int(df["predicted_lung_opacity"].sum()),
    "/",
    len(df),
)

print(
    "Mean actual-box energy:",
    round(
        valid["true_box_energy"].mean() * 100,
        2,
    ),
    "%",
)

print(
    "Mean random-box energy:",
    round(
        valid["random_box_energy"].mean() * 100,
        2,
    ),
    "%",
)

print(
    "Mean improvement:",
    round(
        valid["energy_difference"].mean() * 100,
        2,
    ),
    "percentage points",
)

print(
    "Images with actual > random:",
    int(
        (valid["energy_difference"] > 0).sum()
    ),
    "/",
    len(valid),
)

print("CSV saved:", csv_path)


# ============================================================
# STEP 10: CREATE COMPARISON GRAPH
# ============================================================

means = [
    valid["true_box_energy"].mean() * 100,
    valid["random_box_energy"].mean() * 100,
]

fig, ax = plt.subplots(
    figsize=(7, 5)
)

bars = ax.bar(
    [
        "Radiologist boxes",
        "Random boxes",
    ],
    means,
    color=[
        "#13b8d2",
        "#8290a5",
    ],
)

ax.set_ylabel(
    "Mean Grad-CAM energy inside boxes (%)"
)

ax.set_title(
    "MedVision AI | Validation-Only Grad-CAM Alignment"
)

ax.set_ylim(
    0,
    max(means) * 1.3
)

for bar, value in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f"{value:.2f}%",
        ha="center",
    )

plt.tight_layout()

figure_path = (
    REPORT_DIR
    / "gradcam_random_baseline_validation_100.png"
)

plt.savefig(
    figure_path,
    dpi=180,
)

plt.close(fig)

print("Graph saved:", figure_path)
print("\nExperiment finished successfully!")
