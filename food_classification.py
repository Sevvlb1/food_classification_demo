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


IMG_DIR   = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\all_images"
LABEL_DIR = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\labels"

NUM_CLASSES = 116
BATCH_SIZE  = 2
EPOCHS      = 10
IMG_SIZE    = 416
LR          = 0.005

CONF_THRESHOLD = 0.35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FOOD_MAP = {
    0:   "adanakebap",
    1:   "armut",
    2:   "asure",
    3:   "ayran",
    4:   "baklava",
    5:   "bamya",
    6:   "beyazpeynir",
    7:   "beyazekmek",
    8:   "beyti",
    9:   "bezelye",
    10:  "brokoli",
    11:  "browni",
    12:  "bulgur",
    13:  "cheesecake",
    14:  "cikolatabitter",
    15:  "cikolatasutlu",
    16:  "cigkofte",
    17:  "cilek",
    18:  "cornflex",
    19:  "dolma",
    20:  "domates",
    21:  "donat",
    22:  "dondurma",
    23:  "ekler",
    24:  "elma",         
    25:  "enginar",
    26:  "etsote",
    27:  "ezogelin",         
    28:  "fanta",             
    29:  "firindatavuk",        
    30:  "gozleme",
    31:  "halkatatli",        
    32:  "hamburger",         
    33:  "hamsi",        
    34:  "helva",
    35:  "humus",
    36:  "hunkarbegendi",
    37:  "icelatte",
    38:  "iclikofte",
    39:  "iskender",
    40:  "ispanak",
    41:  "kabak",
    42:  "kadayif",
    43:  "karnabahar",
    44:  "karniyarik",
    45:  "karpuz",
    46:  "kek",
    47:  "kemalpasa",
    48:  "kofte",
    49:  "kola",
    50:  "krep",
    51:  "kumpir",
    52:  "kumru",
    53:  "kunefe",
    54:  "kurabiye",
    55:  "kurufasulye",
    56:  "kuruyemis",
    57:  "lahmacun",
    58:  "lazanya",
    59:  "limonata",
    60:  "makarna",
    61:  "mandalina",
    62:  "mantarsote",
    63:  "biber",
    64:  "cay",
    65:  "zeytin",
    66:  "yogurt",
    67:  "tursu",
    68:  "pilav",
    69:  "marul",
    70:  "limon",
    71:  "sarma",
    72:  "patates",
    73:  "salatalik",
    74:  "ketcap",
    75:  "salata",
    76:  "recel",
    77:  "mayonez",
    78:  "pirzola",
    79:  "pide",
    80:  "manti",
    81:  "menemen",
    82:  "mercimekcorba",
    83:  "meyvesuyu",
    84:  "muz",
    85:  "nohut",
    86:  "tahilliekmek",
    87:  "omlet",
    88:  "pankek",
    89:  "turkkahvesi",
    90:  "pastirma",
    91:  "pizza",
    92:  "pogaca",
    93:  "portakal",
    94:  "sebzegraten",
    95:  "sigaraborek",
    96:  "simit",
    97:  "sinitzel",
    98:  "somon",
    99:  "sucuk",
    100: "sulukofte",
    101: "sut",
    102: "sutlac",
    103: "tantuni",
    104: "tavukdoner",
    105: "tavuksote",
    106: "tavuksuyucorba",
    107: "tazefasulye",
    108: "tiremisu",
    109: "trilece",
    110: "urfa",
    111: "waffle",
    112: "yaglama",
    113: "yulaf",
    114: "yumurtahaslanmis"
    
}


BASE_KCAL = {
    "adanakebap":250, "armut":57, "asure":120, "ayran":37, "baklava":430,
    "bamya":33, "beyazpeynir":264, "beyazekmek":265, "beyti":220, "bezelye":81,
    "brokoli":34, "browni":466, "bulgur":83, "cheesecake":321, "cikolatabitter":546,
    "cikolatasutlu":535, "cigkofte":180, "cilek":32, "cornflex":357, "dolma":150,
    "domates":18, "donat":452, "dondurma":207, "ekler":262, "elma":52,
    "enginar":47, "etsote":200, "ezogelin":80, "fanta":41, "firindatavuk":239,
    "gozleme":250, "halkatatli":300, "hamburger":295, "hamsi":210, "helva":516,
    "humus":166, "hunkarbegendi":180, "icelatte":60, "iclikofte":250, "iskender":220,
    "ispanak":23, "kabak":17, "kadayif":350, "karnabahar":25, "karniyarik":200,
    "karpuz":30, "kek":350, "kemalpasa":280, "kofte":250, "kola":42,
    "krep":227, "kumpir":150, "kumru":300, "kunefe":430, "kurabiye":500,
    "kurufasulye":127, "kuruyemis":600, "lahmacun":250, "lazanya":132, "limonata":40,
    "makarna":131, "mandalina":53, "mantarsote":90, "biber":31, "cay":1,
    "zeytin":115, "yogurt":59, "tursu":11, "pilav":130, "marul":15,
    "limon":29, "sarma":180, "patates":77, "salatalik":16, "ketcap":112,
    "salata":33, "recel":278, "mayonez":680, "pirzola":294, "pide":275,
    "manti":200, "menemen":150, "mercimekcorba":60, "meyvesuyu":45, "muz":89,
    "nohut":164, "tahilliekmek":247, "omlet":154, "pankek":227, "turkkahvesi":2,
    "pastirma":250, "pizza":266, "pogaca":320, "portakal":47, "sebzegraten":120,
    "sigaraborek":300, "simit":275, "sinitzel":250, "somon":208, "sucuk":450,
    "sulukofte":180, "sut":42, "sutlac":111, "tantuni":220, "tavukdoner":215,
    "tavuksote":180, "tavuksuyucorba":50, "tazefasulye":35, "tiremisu":240, "trilece":270,
    "urfa":260, "waffle":291, "yaglama":250, "yulaf":389, "yumurtahaslanmis":155
}

PORTION_G = {
    "adanakebap":200, "armut":150, "asure":200, "ayran":250, "baklava":60,
    "bamya":150, "beyazpeynir":50, "beyazekmek":50, "beyti":200, "bezelye":150,
    "brokoli":150, "browni":60, "bulgur":200, "cheesecake":120, "cikolatabitter":40,
    "cikolatasutlu":40, "cigkofte":100, "cilek":150, "cornflex":40, "dolma":150,
    "domates":100, "donat":70, "dondurma":100, "ekler":80, "elma":180,
    "enginar":200, "etsote":200, "ezogelin":250, "fanta":330, "firindatavuk":200,
    "gozleme":200, "halkatatli":150, "hamburger":200, "hamsi":150, "helva":100,
    "humus":100, "hunkarbegendi":250, "icelatte":300, "iclikofte":150, "iskender":300,
    "ispanak":150, "kabak":150, "kadayif":150, "karnabahar":150, "karniyarik":250,
    "karpuz":300, "kek":80, "kemalpasa":100, "kofte":150, "kola":330,
    "krep":100, "kumpir":300, "kumru":200, "kunefe":150, "kurabiye":50,
    "kurufasulye":250, "kuruyemis":30, "lahmacun":150, "lazanya":250, "limonata":300,
    "makarna":200, "mandalina":100, "mantarsote":150, "biber":100, "cay":200,
    "zeytin":30, "yogurt":150, "tursu":50, "pilav":200, "marul":100,
    "limon":60, "sarma":150, "patates":150, "salatalik":100, "ketcap":20,
    "salata":150, "recel":20, "mayonez":20, "pirzola":200, "pide":250,
    "manti":250, "menemen":200, "mercimekcorba":250, "meyvesuyu":200, "muz":120,
    "nohut":200, "tahilliekmek":50, "omlet":120, "pankek":100, "turkkahvesi":60,
    "pastirma":30, "pizza":200, "pogaca":80, "portakal":180, "sebzegraten":200,
    "sigaraborek":100, "simit":120, "sinitzel":150, "somon":150, "sucuk":50,
    "sulukofte":250, "sut":200, "sutlac":200, "tantuni":200, "tavukdoner":200,
    "tavuksote":200, "tavuksuyucorba":250, "tazefasulye":200, "tiremisu":120, "trilece":150,
    "urfa":200, "waffle":150, "yaglama":100, "yulaf":60, "yumurtahaslanmis":60
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

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.images))

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
