import torch
import cv2
import numpy as np
import glob
import os
from model import build_model
from torchvision import transforms

def load_checkpoint_safe(path, device):
    """
    Try to load checkpoint normally. If it fails due to PyTorch weights-only safety,
    allowlist CrossEntropyLoss and reload with weights_only=False.
    Only do the allowlist if you trust the checkpoint source.
    """
    try:
        # First try the default load (PyTorch 2.6+ may use weights_only=True by default)
        ck = torch.load(path, map_location=device)
        return ck
    except Exception as e:
        print(f"[warning] initial torch.load failed: {e}")
        print("[info] Attempting to allowlist CrossEntropyLoss and reload (only do this if you trust the file).")
        # allowlist the loss class (persist for process) — safer to add only what's necessary
        torch.serialization.add_safe_globals([torch.nn.modules.loss.CrossEntropyLoss])
        ck = torch.load(path, map_location=device, weights_only=False)
        return ck

def main():
    DATA_PATH = '../data/test_images'
    IMAGE_SIZE = 224
    DEVICE = torch.device('cpu')  # change to 'cuda' if available and desired
    class_names = ['clean', 'dirty']

    # Build model (same as during training)
    model = build_model(pretrained=False, fine_tune=False, num_classes=2)
    model.to(DEVICE)

    # Load checkpoint safely
    ck_path = '../outputs/model_pretrained_True.pth'
    checkpoint = load_checkpoint_safe(ck_path, DEVICE)

    # Determine where the state dict is
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # maybe the checkpoint _is_ the state_dict
            state_dict = checkpoint
    else:
        raise RuntimeError("Loaded checkpoint is not a dict or state_dict. Inspect the file manually.")

    # If the state dict keys are prefixed with 'module.' but model expects no prefix, strip it:
    new_state = {}
    model_keys = set(model.state_dict().keys())
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('module.') and not any(k.startswith('module.') for k in model_keys):
        for k, v in state_dict.items():
            new_state[k.replace('module.', '', 1)] = v
        state_dict = new_state

    print('Loading trained model weights...')
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare transforms once
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    all_image_paths = glob.glob(os.path.join(DATA_PATH, "*"))

    os.makedirs("../outputs", exist_ok=True)

    for image_path in all_image_paths:
        # basic image exists check
        image = cv2.imread(image_path)
        if image is None:
            print(f"[warning] could not read {image_path}, skipping.")
            continue
        orig_image = image.copy()

        # Ground truth extraction (keep your existing scheme)
        gt_class_name = os.path.basename(image_path).split('.')[0].split('_')[0]

        # Preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_t = transform(image_rgb)
        img_t = img_t.unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(img_t)                      # tensor on DEVICE
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).cpu().item())
            pred_class_name = class_names[pred_idx]

        print(f"GT: {gt_class_name}, Pred: {pred_class_name}")

        # Annotate
        cv2.putText(orig_image, f"GT: {gt_class_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        cv2.putText(orig_image, f"Pred: {pred_class_name}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA)

        # Show and save
        cv2.imshow('Result', orig_image)
        cv2.waitKey(0)

        # Avoid overwriting files: include original basename
        out_name = f"{gt_class_name}_{os.path.basename(image_path)}"
        out_path = os.path.join("../outputs", out_name)
        cv2.imwrite(out_path, orig_image)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
