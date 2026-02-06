# scripts/3_model_training.py

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import (
    MODEL_DIR, MODEL_NAME, SCALER_NAME,
    TEST_SIZE, RANDOM_STATE, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, PATIENCE
)

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 1. LOAD DATA SAFELY (IMPORTANT)
# -------------------------------
def load_data():
    X, y = [], []
    data_root = "data"

    for folder in os.listdir(data_root):
        path = os.path.join(data_root, folder)
        if not os.path.isdir(path):
            continue
        
        try:
            label = int(folder.split('_')[0])  # folder like "0_A"
        except:
            continue

        print(f"Loading folder: {folder}")

        for file in os.listdir(path):
            if file.endswith(".npy"):
                file_path = os.path.join(path, file)
                arr = np.load(file_path)

                # MUST BE LENGTH 63
                if arr.shape == (63,):
                    X.append(arr)
                    y.append(label)
                else:
                    print(f"❌ Skipped corrupted file: {file_path}, shape={arr.shape}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nLoaded {len(X)} valid samples.")
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
# 4. TRAINING FUNCTION
# -------------------------------
def train_model():
    # Load & validate dataset
    X, y = load_data()
    if len(X) == 0:
        print("❌ ERROR: No valid training data found!")
        return

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    joblib.dump(scaler, f"{MODEL_DIR}/{SCALER_NAME}")

    # Torch datasets
    train_ds = GestureDataset(X_train, y_train)
    val_ds = GestureDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X_train.shape[1], num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    print("\n🔵 Training started...\n")

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(1, EPOCHS + 1):

        # TRAIN
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # VALIDATE
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS} — Val Accuracy: {acc:.4f}")

        # EARLY STOPPING
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("\n🟡 Early stopping triggered.")
                break

    print(f"\n✅ Training complete! Best Accuracy: {best_acc:.4f}")
    print(f"📁 Model saved to: {MODEL_DIR}/{MODEL_NAME}")
    print(f"📁 Scaler saved to: {MODEL_DIR}/{SCALER_NAME}")


# -------------------------------
# RUN TRAINING
# -------------------------------
if __name__ == "__main__":
    train_model()
