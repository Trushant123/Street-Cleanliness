# streamlit_app_pytorch.py
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from model import build_model  # your build_model function

st.set_page_config(layout="wide")
st.title("Street Cleanliness Classifier (PyTorch)")
st.markdown("Upload an image and the model will classify it as **Clean** or **Dirty** and show **why** (Grad-CAM).")

# -----------------------
# Config
# -----------------------
MODEL_PATH = "../outputs/model_pretrained_True.pth"  # adjust if necessary
IMAGE_SIZE = 224
CLASS_NAMES = ["Clean", "Dirty"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Utilities: safe checkpoint loader
# -----------------------
def load_checkpoint_safe(path, device):
    """
    Try to torch.load checkpoint. If it fails due to PyTorch weights_only safety,
    allowlist CrossEntropyLoss and reload with weights_only=False.
    """
    try:
        ck = torch.load(path, map_location=device)
        return ck
    except Exception as e:
        # Attempt allowlist only if the error mentions weights_only / pickling safety
        errmsg = str(e)
        if "Weights only load failed" in errmsg or "WeightsUnpicklingError" in errmsg or "weights_only" in errmsg:
            torch.serialization.add_safe_globals([torch.nn.modules.loss.CrossEntropyLoss])
            ck = torch.load(path, map_location=device, weights_only=False)
            return ck
        else:
            raise

# -----------------------
# Model loader (cached)
# -----------------------
@st.cache_resource
def load_pytorch_model(model_path: str, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Build model architecture (must match training)
    model = build_model(pretrained=False, fine_tune=False, num_classes=len(CLASS_NAMES))
    model.to(device)

    # Load checkpoint safely
    checkpoint = load_checkpoint_safe(model_path, device)

    # Resolve state dict location
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            # assume checkpoint _is_ state_dict
            state = checkpoint
    else:
        raise RuntimeError("Loaded checkpoint is not a state dict or dict with state dict keys.")

    # handle 'module.' prefix if present
    new_state = {}
    first_key = next(iter(state.keys()))
    if first_key.startswith("module.") and not any(k.startswith("module.") for k in model.state_dict().keys()):
        for k, v in state.items():
            new_state[k.replace("module.", "", 1)] = v
        state = new_state

    model.load_state_dict(state)
    model.eval()
    return model

# -----------------------
# Preprocess
# -----------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# Grad-CAM helpers
# -----------------------
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = self._find_target_layer()
        if self.target_layer is None:
            raise RuntimeError("No Conv2d layer found in the model for Grad-CAM.")

        # register hooks
        self._register_hooks()

    def _find_target_layer(self):
        # Find the last nn.Conv2d layer in the model
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; take the first element
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # backward_hook API: register_full_backward_hook is preferred on newer PyTorch, but register_backward_hook works for most
        try:
            self.target_layer.register_backward_hook(backward_hook)
        except Exception:
            # Fallback for newer PyTorch versions:
            self.target_layer.register_full_backward_hook(lambda module, grad_input, grad_output: backward_hook(module, grad_input, grad_output))

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: torch tensor shape [1, C, H, W]
        class_idx: index of target class. If None, uses predicted class.
        returns: heatmap numpy array (H, W) in [0,1]
        """
        self.model.zero_grad()
        outputs = self.model(input_tensor)  # [1, num_classes]
        if class_idx is None:
            class_idx = int(outputs.argmax(dim=1).item())
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        # gradients: [N, C, H, W]  -> take global average pool over H,W
        grads = self.gradients  # [N, C, H, W]
        activations = self.activations  # [N, C, H, W]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        weighted_activations = weights * activations  # [N, C, H, W]
        cam = weighted_activations.sum(dim=1, keepdim=True)  # [N, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Normalize to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap_cv(original_img, heatmap, alpha=0.4):
    # heatmap: HxW (0..1), original_img: HxW(x3) RGB uint8
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # convert heatmap_color BGR->RGB since applyColorMap returns BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

# -----------------------
# Load model (cached)
# -----------------------
try:
    model = load_pytorch_model(MODEL_PATH, DEVICE)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------
# UI: file uploader
# -----------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload an image to get started.")
    st.stop()

# Read image into numpy RGB
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Couldn't read the image. Try a different file.")
    st.stop()
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

st.image(rgb, caption="Uploaded Image", use_container_width=True)

# Preprocess and infer
input_tensor = preprocess(rgb)  # C,H,W in float tensor
input_batch = input_tensor.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(input_batch)  # logits
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # numpy array
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

st.success(f"### Prediction: {pred_label}")
st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

# Confidence bar
fig, ax = plt.subplots(figsize=(6, 1.6))
ax.barh(CLASS_NAMES, probs, edgecolor='k')
ax.set_xlim([0, 1])
ax.set_xlabel("Confidence")
for i, v in enumerate(probs):
    ax.text(v + 0.01, i, f"{v * 100:.2f}%", va="center", fontweight="bold")
plt.tight_layout()
st.pyplot(fig)

# Grad-CAM visualization (requires gradient)
try:
    gradcam = GradCAM(model)
    # Need to run forward again but with grad enabled
    input_batch = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    cam = gradcam(input_batch, class_idx=pred_idx)  # HxW normalized
    overlay = overlay_heatmap_cv(rgb, cam, alpha=0.5)
    st.markdown("### Grad-CAM: Regions influencing the prediction")
    st.image(overlay, use_container_width=True)
except Exception as e:
    st.warning(f"Grad-CAM failed: {e}")
