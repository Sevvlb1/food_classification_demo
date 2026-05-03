import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR


IMG_DIR   = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\all_images_new"
LABEL_DIR = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\labels22"

BATCH_SIZE  = 2
EPOCHS      = 10
IMG_SIZE    = 416
LR          = 0.001

CONF_THRESHOLD = 0.35
#testte threshold 0.2 dene

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FOOD_MAP = {
    0: "baklagil",
    1: "ekmek",
    2: "pilav",
    3: "kirmizi et",
    4: "salata",
    5: "balik",
    6: "patates",
    7: "tavuk",
    8: "sebze",
    9: "makarna",
    10: "corba",
    11: "zeytinyagli", 
    12: "yumurta",
    13: "yogurt",
    14: "meyve",
    15: "manti",
    16: "pide",
    17: "fastfood",
    18: "lahmacun",
    19: "tatli"
}
NUM_CLASSES = len(FOOD_MAP) + 1


BASE_KCAL = {
    "baklagil":250, "ekmek":57, "pilav":120, "kirmizi et":37, "salata":430,
    "balik":250, "patates":57, "tavuk":120, "sebze":37, "makarna": 150,
    "corba": 100, "zeytinyagli": 100, "yumurta": 150, "yogurt": 40,
    "meyve": 70, "manti": 100, "fastfood": 500, "lahmacun": 150, "tatli": 250

}

PORTION_G = {
    "baklagil":200, "ekmek":150, "pilav":200, "kirmizi et":250, "salata":60,
    "balik":250, "patates":57, "tavuk":120, "sebze":37, "makarna": 150,
    "corba": 100, "zeytinyagli": 100, "yumurta": 150, "yogurt": 40,
    "meyve": 70, "manti": 100, "fastfood": 500, "lahmacun": 150, "tatli": 250

}


DEFAULT_KCAL    = 200
DEFAULT_PORTION = 150


def estimate_calories(yolo_class_id, box, img_size=416):
    food_name     = FOOD_MAP.get(yolo_class_id, None)

    if food_name is None:
        print(f"  WARNING: yolo_class_id {yolo_class_id} not in FOOD_MAP — using defaults")
        food_name = f"unknown_{yolo_class_id}"

    kcal_per_100g = BASE_KCAL.get(food_name, DEFAULT_KCAL)
    portion_g     = PORTION_G.get(food_name, DEFAULT_PORTION)

    x1, y1, x2, y2 = box
    box_area   = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = img_size * img_size
    area_ratio = box_area / frame_area

    
    reference_ratio = 0.20
    scale           = max(0.3, min(area_ratio / reference_ratio, 3.0))
    weight_g        = portion_g * scale

    calories = (weight_g / 100.0) * kcal_per_100g
    return round(calories, 1), round(weight_g, 1), food_name



def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)



def yolo_to_pixel(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]



class FoodDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.tf        = transforms.ToTensor()

        all_images = os.listdir(img_dir)
        self.images = []
        skipped = 0

        for img_name in all_images:
            label_path = self._label_path(img_name)
            if not os.path.exists(label_path):
                skipped += 1
                continue
            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                skipped += 1
                continue
            self.images.append(img_name)

        print(f"Dataset: {len(self.images)} valid | {skipped} skipped (no labels)")

    def _label_path(self, img_name):
        base = os.path.splitext(img_name)[0]
        return os.path.join(self.label_dir, base + ".txt")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name   = self.images[idx]
        img_path   = os.path.join(self.img_dir, img_name)
        label_path = self._label_path(img_name)

        img = imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.images))

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls, x, y, bw, bh = map(float, parts)
                x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, IMG_SIZE, IMG_SIZE)

                x1 = max(0.0, min(x1, IMG_SIZE - 1))
                y1 = max(0.0, min(y1, IMG_SIZE - 1))
                x2 = max(0.0, min(x2, IMG_SIZE - 1))
                y2 = max(0.0, min(y2, IMG_SIZE - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # 0 = background

        '''if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.images))'''

        if len(boxes) == 0:
            return None
        

        boxes   = torch.as_tensor(boxes,  dtype=torch.float32)
        labels  = torch.as_tensor(labels, dtype=torch.int64)
        areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        return self.tf(img), {
            "boxes":   boxes,
            "labels":  labels,
            "area":    areas,
            "iscrowd": iscrowd,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))



def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model



def train():
    dataset = FoodDataset(IMG_DIR, LABEL_DIR)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    model     = get_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005
    )
    scheduler    = StepLR(optimizer, step_size=3, gamma=0.1)
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0, 0

        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += losses.item()
                n_batches  += 1
            except Exception as e:
                print(f"  Skipped batch: {e}")
                continue

        scheduler.step()
        torch.cuda.empty_cache()

        avg = total_loss / max(n_batches, 1)
        loss_history.append(avg)
        print(f"Epoch {epoch:>2}/{EPOCHS} | Total: {total_loss:.4f} | Avg: {avg:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved → model.pth")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o", color="steelblue")
    plt.xlabel("Epoch"); plt.ylabel("Avg Loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    plt.show()



def load_model():
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, img_path, threshold=CONF_THRESHOLD):
    img = imread(img_path)
    if img is None:
        print("Image could not be read.")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor      = transforms.ToTensor()(img_rgb).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    boxes  = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    print(f"\nTotal raw predictions: {len(boxes)}")
    if len(scores):
        print(f"Score range: {scores.min():.3f} – {scores.max():.3f}")
    print(f"Applying threshold: {threshold}")

    canvas     = img_rgb.copy()
    total_kcal = 0.0
    legend     = []

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        yolo_id = int(label) - 1  # convert back from 1-based to 0-based
        kcal, weight_g, food_name = estimate_calories(yolo_id, box, IMG_SIZE)
        total_kcal += kcal

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)

        tag = f"{food_name} {score:.2f} ~{kcal:.0f}kcal"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(y1-th-8, 0)), (x1+tw, max(y1, th+8)), (0,180,0), -1)
        cv2.putText(canvas, tag, (x1, max(y1-4, th+4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        legend.append(
            f"{food_name:20s}  score={score:.2f}  "
            f"~{weight_g:.0f}g  →  {kcal:.0f} kcal"
        )

    if not legend:
        print(f"\nNo detections above threshold {threshold}.")
        print("Try lowering CONF_THRESHOLD at the top of the script (e.g. 0.20).")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(
        f"Toplam Tahmini Kalori: {total_kcal:.0f} kcal",
        fontsize=14, fontweight="bold"
    )

    if legend:
        ax.text(
            0.01, 0.01, "\n".join(legend),
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88),
        )

    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n--- Kalori Özeti ---")
    for line in legend:
        print(" ", line)
    print(f"\n  TOPLAM: {total_kcal:.0f} kcal")



if __name__ == "__main__":
    print("1 - Eğit (Train)")
    print("2 - Test (Predict)")
    choice = input("Seçim: ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        mdl  = load_model()
        path = input("Görüntü yolu (Image path): ").strip()
        predict(mdl, path)
    else:
        print("Geçersiz seçim.")
