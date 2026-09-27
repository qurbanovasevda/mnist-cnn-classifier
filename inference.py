import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from models import CNN

# --- Model yükləmə ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
model.to(device)
model.eval()

print("Model uğurla yükləndi (best_cnn.pth-dan)\n")

# --- Preprocessing funksiyası ---
def load_and_preprocess(image_path):
    # Şəkli oxu, grayscale et
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert et (fon qara, rəqəm ağ olsun - MNIST formatı)
    img = 255 - img

    # Threshold - səs-küyü təmizlə (aşağı intensivliyi sıfırla)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)

    # Rəqəmin ətrafındakı boş sahəni tap və kəs (bounding box)
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Kvadrat şəklə sal (uzun tərəfə görə), mərkəzləşdir
    size = max(w, h)
    padded = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = digit

    # Ətrafına əlavə boşluq qoy (MNIST-də rəqəm tam kənara qədər getmir)
    border = size // 5
    final_size = size + 2 * border
    bordered = np.zeros((final_size, final_size), dtype=np.uint8)
    bordered[border:border+size, border:border+size] = padded

    # 28x28-ə kiçilt
    resized = cv2.resize(bordered, (28, 28), interpolation=cv2.INTER_AREA)

    # Xəttləri bir az qalınlaşdır (dilate)
    kernel = np.ones((2, 2), np.uint8)
    resized = cv2.dilate(resized, kernel, iterations=1)

    # PyTorch tensoruna çevir və normallaşdır
    img_tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    img_tensor = (img_tensor - 0.1307) / 0.3081
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)

    return img_tensor

# --- Custom şəkilləri tap ---
image_paths = sorted(glob.glob("custom_digits/*.png") + glob.glob("custom_digits/*.jpg") + glob.glob("custom_digits/*.jpeg"))

if len(image_paths) == 0:
    print("XƏTA: custom_digits/ qovluğunda şəkil tapılmadı!")
else:
    print(f"{len(image_paths)} şəkil tapıldı: {image_paths}\n")

    fig, axes = plt.subplots(1, len(image_paths), figsize=(3 * len(image_paths), 3))
    if len(image_paths) == 1:
        axes = [axes]

    for i, path in enumerate(image_paths):
        img_tensor = load_and_preprocess(path).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        pred_digit = predicted.item()
        conf_value = confidence.item() * 100

        print(f"{os.path.basename(path)} -> Proqnoz: {pred_digit} (Əminlik: {conf_value:.1f}%)")

        # Şəkli göstər (preprocessed hal, yəni model gördüyü kimi)
        img_display = img_tensor.squeeze().cpu().numpy()
        axes[i].imshow(img_display, cmap="gray")
        axes[i].set_title(f"Pred: {pred_digit}\n({conf_value:.1f}%)")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("custom_digit_predictions.png")
    print("\nSaved custom_digit_predictions.png")
    plt.show()