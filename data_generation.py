import glob
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== ПОЛНЫЙ РУССКИЙ АЛФАВИТ ====================

ALPHABET_UPPER = list(string.ascii_uppercase)


def find_fonts():
    font_paths = []

    search_dirs = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ]
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ttf in glob.glob(
                os.path.join(search_dir, "**", "*.ttf"), recursive=True
            ):
                if "Arial" in ttf:
                    font_paths.append(ttf)

    return font_paths


def generate_letter_image(
    letter,
    img_size=28,
    font_size=20,
    noise_level=0.05,
    shift_range=2,
    rotation_range=10,
    font_path=None,
    thickness_variation=True,
):
    if thickness_variation:
        actual_font_size = font_size + random.randint(-3, 3)
        actual_font_size = max(12, actual_font_size)
    else:
        actual_font_size = font_size

    font = ImageFont.truetype(font_path, actual_font_size)

    canvas_size = int(img_size * 1.5)
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    x = (canvas_size - text_w) // 2 - bbox[0] + dx
    y = (canvas_size - text_h) // 2 - bbox[1] + dy

    draw.text((x, y), letter, fill=255, font=font)

    angle = random.uniform(-rotation_range, rotation_range)
    img = img.rotate(angle, fillcolor=0)

    left = (canvas_size - img_size) // 2
    top = (canvas_size - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))

    img_array = np.array(img, dtype=np.float32) / 255.0

    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_array.shape)
        img_array = np.clip(img_array + noise, 0.0, 1.0)

    return img_array


def generate_dataset(
    letters,
    samples_per_letter=5,
    img_size=28,
    font_size=20,
    noise_level=0.05,
    shift_range=2,
    rotation_range=10,
    font_paths=None,
):
    if font_paths is None:
        font_paths = find_fonts()

    X = []
    y = []
    label_map = {i: letter for i, letter in enumerate(letters)}

    total = len(letters) * samples_per_letter
    print(
        f"\nГенерация датасета: {len(letters)} букв × "
        f"{samples_per_letter} примеров = {total} изображений"
    )

    for class_idx, letter in enumerate(letters):
        success_count = 0
        attempts = 0
        max_attempts = samples_per_letter * 3

        while success_count < samples_per_letter and attempts < max_attempts:
            attempts += 1
            fp = random.choice(font_paths)

            img = generate_letter_image(
                letter,
                img_size=img_size,
                font_size=font_size,
                noise_level=noise_level,
                shift_range=shift_range,
                rotation_range=rotation_range,
                font_path=fp,
            )

            if img.max() > 0.1:
                X.append(img.flatten())
                y.append(class_idx)
                success_count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    print(f"\nДатасет создан: {X.shape[0]} примеров, " f"размер входа: {X.shape[1]}")

    return X, y, label_map


# ==================== ВИЗУАЛИЗАЦИЯ ====================


def visualize_samples(X, y, label_map, n_per_class=3, img_size=28):
    """Показывает примеры изображений для каждой буквы."""
    import matplotlib.pyplot as plt

    n_classes = len(label_map)
    fig, axes = plt.subplots(
        n_classes, n_per_class, figsize=(n_per_class * 1.5, n_classes * 1.5)
    )

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx in range(n_classes):
        class_indices = np.where(y == class_idx)[0]
        for j in range(n_per_class):
            if j < len(class_indices):
                idx = class_indices[j]
                img = X[idx].reshape(img_size, img_size)
                axes[class_idx, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[class_idx, j].axis("off")
            if j == 0:
                axes[class_idx, j].set_ylabel(
                    label_map[class_idx],
                    rotation=0,
                    fontsize=12,
                    labelpad=20,
                    va="center",
                )

    plt.suptitle("Примеры сгенерированных изображений", fontsize=14)
    plt.tight_layout()
    plt.savefig("./samples_all_letters.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print(f"\nАлфавит ({len(ALPHABET_UPPER)} букв):")
    print(" ".join(ALPHABET_UPPER))

    X, y, label_map = generate_dataset(
        letters=ALPHABET_UPPER,
        samples_per_letter=50,
        img_size=28,
        font_size=20,
        noise_level=0.05,
        shift_range=2,
        rotation_range=10,
    )

    visualize_samples(X, y, label_map, n_per_class=5)
