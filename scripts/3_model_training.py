"""
3_model_training.py
Trains the MLP gesture classifier on collected landmark data.

Improvements over a basic training loop:
  - Handles partial data collection gracefully (trains on whichever classes
    you've actually collected so far, not a hardcoded class count)
  - Saves a label_map.json so inference scripts always match what the
    model was actually trained on, even if config.py changes later
  - Saves a classification report (precision/recall/F1 per class) and a
    confusion matrix image - the actual numbers you need for your resume
    and for debugging which letters get confused with each other
"""

import os
import json

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # headless - just save to file, no display needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    MODEL_DIR, MODEL_NAME, SCALER_NAME, METRICS_NAME, CONFUSION_MATRIX_NAME,
    LABEL_MAP_NAME, GESTURE_LABELS, TEST_SIZE, RANDOM_STATE, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, PATIENCE
)

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------
# 1. LOAD DATA SAFELY
# -------------------------------
def load_data():
    X, y = [], []
    data_root = "data"

    for folder in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, folder)
        if not os.path.isdir(path):
            continue

        try:
            label = int(folder.split('_')[0])
        except (ValueError, IndexError):
            continue

        print(f"Loading folder: {folder}")
        loaded_here = 0
        for file in os.listdir(path):
            if file.endswith(".npy"):
                file_path = os.path.join(path, file)
                arr = np.load(file_path)
                if arr.shape == (63,):
                    X.append(arr)
                    y.append(label)
                    loaded_here += 1
                else:
                    print(f"❌ Skipped corrupted file: {file_path}, shape={arr.shape}")
        print(f"   -> {loaded_here} valid samples")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"\nLoaded {len(X)} valid samples total across {len(set(y.tolist()))} classes.")
    return X, y


# -------------------------------
# 2. DATASET CLASS
# -------------------------------
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------
# 3. MODEL (MLP)
# -------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# 4. EVALUATION + ARTIFACT SAVING
# -------------------------------
def evaluate_and_save_artifacts(model, X_val, y_val, device, label_names):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds = torch.argmax(model(x_tensor), dim=1).cpu().numpy()

    report_dict = classification_report(
        y_val, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_val, preds, target_names=label_names, zero_division=0
    )
    print("\n📊 Classification Report:\n")
    print(report_text)

    with open(os.path.join(MODEL_DIR, METRICS_NAME), "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"📁 Metrics saved to: {MODEL_DIR}/{METRICS_NAME}")

    cm = confusion_matrix(y_val, preds)
    size = max(6, len(label_names) * 0.4)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, CONFUSION_MATRIX_NAME), dpi=150)
    plt.close(fig)
    print(f"📁 Confusion matrix saved to: {MODEL_DIR}/{CONFUSION_MATRIX_NAME}")


# -------------------------------
# 5. TRAINING FUNCTION
# -------------------------------
def train_model():
    X, y = load_data()
    if len(X) == 0:
        print("❌ ERROR: No valid training data found!")
        return

    present_classes = sorted(set(y.tolist()))
    label_names = [GESTURE_LABELS[c] for c in present_classes]
    num_classes = len(present_classes)
    print(f"Classes present in data ({num_classes}): {label_names}")

    if num_classes < len(GESTURE_LABELS):
        missing = [GESTURE_LABELS[i] for i in GESTURE_LABELS if i not in present_classes]
        print(f"⚠️  Note: training on a subset. Missing classes: {missing}")

    # Remap original label indices to a contiguous 0..num_classes-1 range,
    # since some gestures may not have been collected yet.
    remap = {orig: i for i, orig in enumerate(present_classes)}
    y_remapped = np.array([remap[v] for v in y], dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_remapped, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_remapped
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))

    # Save the label mapping actually used for THIS trained model.
    with open(os.path.join(MODEL_DIR, LABEL_MAP_NAME), "w") as f:
        json.dump({str(i): GESTURE_LABELS[orig] for orig, i in remap.items()}, f, indent=2)

    train_ds = GestureDataset(X_train, y_train)
    val_ds = GestureDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    print("\n🔵 Training started...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = torch.argmax(model(xb), dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS} — Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\n🟡 Early stopping triggered.")
                break

    print(f"\n✅ Training complete! Best Accuracy: {best_acc:.4f}")
    print(f"📁 Model saved to: {MODEL_DIR}/{MODEL_NAME}")
    print(f"📁 Scaler saved to: {MODEL_DIR}/{SCALER_NAME}")

    # Reload the best checkpoint (not just the last epoch) for final evaluation.
    best_model = MLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    best_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=device))
    evaluate_and_save_artifacts(best_model, X_val, y_val, device, label_names)


if __name__ == "__main__":
    train_model()
