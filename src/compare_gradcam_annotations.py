
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ID = "00436515-870c-4b36-a041-de91049b9ab4"

image_path = ROOT / "data" / "processed" / "images_224" / f"{IMAGE_ID}.png"
labels_path = ROOT / "data" / "raw" / "stage_2_train_labels.csv"
model_path = ROOT / "checkpoints" / "resnet18_stage2_best.pth"
output_path = ROOT / "reports" / "annotation_gradcam_comparison.png"

image = Image.open(image_path).convert("RGB")
annotations = pd.read_csv(labels_path)
boxes = annotations[
    (annotations["patientId"] == IMAGE_ID) &
    (annotations["Target"] == 1)
]

checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

names = [
    name for name, index in sorted(
        checkpoint["class_to_idx"].items(),
        key=lambda item: item[1]
    )
]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.485, 0.485],
        std=[0.229, 0.229, 0.229],
    ),
])

tensor = preprocess(image).unsqueeze(0)
captured = {}

def forward_hook(module, inputs, output):
    captured["activations"] = output
    output.register_hook(
        lambda grad: captured.update(gradients=grad)
    )

handle = model.layer4[-1].register_forward_hook(forward_hook)

try:
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        scores = model(tensor)
        predicted_index = scores.argmax(dim=1).item()
        scores[0, predicted_index].backward()

    activations = captured["activations"].detach()
    gradients = captured["gradients"].detach()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1))[0]
    cam = cam.cpu().numpy()

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)

    cam = np.asarray(cam_image, dtype=np.float32) / 255.0
finally:
    handle.remove()

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(image)
axes[0].set_title("Original X-ray")

axes[1].imshow(image)

# RSNA coordinates are in the original 1024x1024 image space.
# Convert coordinates to the processed image dimensions.
scale_x = image.width / 1024
scale_y = image.height / 1024

for _, box in boxes.iterrows():
    rectangle = Rectangle(
        (box["x"] * scale_x, box["y"] * scale_y),
        box["width"] * scale_x,
        box["height"] * scale_y,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[1].add_patch(rectangle)

axes[1].set_title(f"Radiologist Annotations ({len(boxes)} boxes)")

axes[2].imshow(image)
axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
axes[2].set_title(f"Grad-CAM: {names[predicted_index]}")

for ax in axes:
    ax.axis("off")

fig.suptitle(
    "MedVision AI | Annotation vs Model Attribution",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("Comparison completed!")
print("Predicted class:", names[predicted_index])
print("Radiologist boxes:", len(boxes))
print("Saved to:", output_path)
