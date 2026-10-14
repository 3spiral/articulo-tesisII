from PIL import Image
import pandas as pd

from src.settings import (
    BEHSOF_IMAGES_DIR,
    BEHSOF_INVENTORY_CSV,
)


ROOT = BEHSOF_IMAGES_DIR
OUTPUT = BEHSOF_INVENTORY_CSV

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff"
}

VIDEO_EXTENSIONS = {
    ".avi", ".mp4", ".mov"
}

rows = []

for patient_folder in ROOT.iterdir():

    if not patient_folder.is_dir():
        continue

    patient_id = patient_folder.name
    center = patient_id[:3]

    for file_path in patient_folder.rglob("*"):

        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        file_type = "other"
        width = None
        height = None
        mode = None
        readable = True

        if ext in IMAGE_EXTENSIONS:
            file_type = "image"

            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    img.verify()

            except Exception:
                file_type = "invalid_image"
                readable = False

        elif ext in VIDEO_EXTENSIONS:
            file_type = "video"

        rows.append({
            "patient_id": patient_id,
            "center": center,
            "relative_path": str(file_path.relative_to(ROOT)),
            "filename": file_path.name,
            "extension": ext,
            "file_type": file_type,
            "width": width,
            "height": height,
            "mode": mode,
            "size_bytes": file_path.stat().st_size,
            "readable": readable
        })

df = pd.DataFrame(rows)

df = df.sort_values(
    ["patient_id", "relative_path"]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== RESUMEN ==========")

print("Pacientes:",
      df["patient_id"].nunique())

print("Archivos totales:",
      len(df))

print("\nTipos de archivo:")
print(df["file_type"].value_counts())

print("\nExtensiones:")
print(df["extension"].value_counts())

images = df[
    df["file_type"] == "image"
]

print("\nImágenes totales:",
      len(images))

print("\nImágenes por paciente:")
print(
    images.groupby("patient_id")
          .size()
          .describe()
)

print("\nPacientes con más imágenes:")
print(
    images.groupby("patient_id")
          .size()
          .sort_values(ascending=False)
          .head(20)
)

print("\nResoluciones:")
print(
    images.groupby(["width", "height"])
          .size()
          .sort_values(ascending=False)
          .head(30)
)

print("\nVideos:")
print(
    df[df["file_type"] == "video"]
    [["patient_id", "relative_path"]]
)