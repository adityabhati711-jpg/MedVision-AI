
from pathlib import Path
import torch
from torchvision.models import resnet18

model_path = Path("checkpoints/resnet18_stage2_best.pth")

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("MODEL LOADED SUCCESSFULLY!")
print("Architecture:", checkpoint["architecture"])
print("Classes:", checkpoint["class_to_idx"])
print("Model ready for predictions!")
