import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.settings import (
    BEHSOF_IMAGES_DIR,
    BEHSOF_QUALITY_AUDIT_CSV,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = BEHSOF_IMAGES_DIR
OUTPUT = BEHSOF_QUALITY_AUDIT_CSV

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# UTILIDADES
# ============================================================

def load_gray(path):
    return cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )


def shannon_entropy(gray):
    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    total = hist.sum()

    if total == 0:
        return 0.0

    p = hist / total
    p = p[p > 0]

    return float(
        -np.sum(
            p * np.log2(p)
        )
    )


def get_analysis_roi(gray):
    """
    Recorta zonas periféricas donde suele haber:
    - encabezados
    - hospital
    - nombre/edad/fecha
    - parámetros
    - barras laterales

    NO pretende aislar perfectamente el hígado.
    Solo crea una región más útil que la captura completa.
    """

    h, w = gray.shape

    # Recorte conservador.
    # Evitamos extremos, pero mantenemos la mayor parte
    # del campo ecográfico.
    top = int(h * 0.10)
    bottom = int(h * 0.90)

    left = int(w * 0.10)
    right = int(w * 0.90)

    roi = gray[
        top:bottom,
        left:right
    ]

    return roi


# ============================================================
# LOW INFORMATION
# ============================================================

def analyze_information(gray):

    roi = get_analysis_roi(gray)

    mean_intensity = float(
        np.mean(roi)
    )

    std_intensity = float(
        np.std(roi)
    )

    entropy = shannon_entropy(
        roi
    )

    # píxeles con señal significativa
    ratio_gt_15 = float(
        np.mean(
            roi > 15
        )
    )

    ratio_gt_30 = float(
        np.mean(
            roi > 30
        )
    )

    ratio_gt_45 = float(
        np.mean(
            roi > 45
        )
    )

    # Bordes = aproximación a textura/estructura
    edges = cv2.Canny(
        roi,
        30,
        90
    )

    edge_ratio = float(
        np.mean(
            edges > 0
        )
    )

    # Regla mucho más conservadora.
    #
    # No basta con "imagen oscura".
    # Debe presentar simultáneamente:
    # - muy poca señal
    # - poca entropía
    # - muy poca estructura
    low_information_candidate = (
        (
            ratio_gt_30 < 0.08
            and
            entropy < 2.3
        )
        or
        (
            ratio_gt_15 < 0.12
            and
            edge_ratio < 0.015
        )
    )

    return {
        "roi_mean_intensity":
            mean_intensity,

        "roi_std_intensity":
            std_intensity,

        "roi_entropy":
            entropy,

        "roi_ratio_gt15":
            ratio_gt_15,

        "roi_ratio_gt30":
            ratio_gt_30,

        "roi_ratio_gt45":
            ratio_gt_45,

        "roi_edge_ratio":
            edge_ratio,

        "low_information_candidate":
            low_information_candidate
    }


# ============================================================
# DUAL PANEL
# ============================================================

def analyze_dual_panel(gray):
    """
    Detecta posible dual-panel mediante:

    contenido a la izquierda
    + caída de contenido central
    + contenido a la derecha

    La métrica relevante es center_drop_ratio.
    """

    h, w = gray.shape

    # Ignoramos encabezado y parte inferior
    y1 = int(h * 0.10)
    y2 = int(h * 0.88)

    roi = gray[
        y1:y2,
        :
    ]

    # Dividimos horizontalmente en regiones.
    left = roi[
        :,
        int(w * 0.12):
        int(w * 0.42)
    ]

    center = roi[
        :,
        int(w * 0.44):
        int(w * 0.56)
    ]

    right = roi[
        :,
        int(w * 0.58):
        int(w * 0.88)
    ]

    # Proporción de píxeles con información.
    #
    # 25 funciona mejor que evaluar negro absoluto.
    left_content = float(
        np.mean(
            left > 25
        )
    )

    center_content = float(
        np.mean(
            center > 25
        )
    )

    right_content = float(
        np.mean(
            right > 25
        )
    )

    side_reference = min(
        left_content,
        right_content
    )

    if side_reference > 0:
        center_drop_ratio = (
            center_content /
            side_reference
        )
    else:
        center_drop_ratio = 999.0

    # También comprobamos brillo relativo
    left_mean = float(
        np.mean(left)
    )

    center_mean = float(
        np.mean(center)
    )

    right_mean = float(
        np.mean(right)
    )

    side_mean_reference = min(
        left_mean,
        right_mean
    )

    if side_mean_reference > 0:
        center_mean_ratio = (
            center_mean /
            side_mean_reference
        )
    else:
        center_mean_ratio = 999.0

    # Heurística conservadora:
    #
    # necesitamos contenido suficiente
    # en ambos lados y caída clara central
    dual_panel_candidate = (
        left_content > 0.20
        and
        right_content > 0.20
        and
        center_drop_ratio < 0.58
        and
        center_mean_ratio < 0.72
    )

    return {
        "left_content_ratio":
            left_content,

        "center_content_ratio":
            center_content,

        "right_content_ratio":
            right_content,

        "center_drop_ratio":
            center_drop_ratio,

        "left_mean_intensity":
            left_mean,

        "center_mean_intensity":
            center_mean,

        "right_mean_intensity":
            right_mean,

        "center_mean_ratio":
            center_mean_ratio,

        "dual_panel_candidate_v2":
            dual_panel_candidate
    }


# ============================================================
# IMAGEN COMPLETA
# ============================================================

def inspect_image(path):

    row = {}

    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

        row["width"] = width
        row["height"] = height
        row["mode"] = mode
        row["readable"] = True

    except Exception:
        row["readable"] = False
        return row

    gray = load_gray(path)

    if gray is None:
        row["readable"] = False
        return row

    row.update(
        analyze_information(
            gray
        )
    )

    row.update(
        analyze_dual_panel(
            gray
        )
    )

    row["standard_resolution"] = (
        width == 1024
        and
        height == 768
    )

    return row


# ============================================================
# AUDITORÍA
# ============================================================

rows = []

for patient_folder in sorted(
    [
        x
        for x in ROOT.iterdir()
        if x.is_dir()
    ]
):

    patient_id = (
        patient_folder.name
    )

    center = (
        patient_id[:3]
    )

    for path in patient_folder.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        row = {
            "patient_id":
                patient_id,

            "center":
                center,

            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),

            "filename":
                path.name,

            "extension":
                path.suffix.lower()
        }

        metrics = inspect_image(
            path
        )

        row.update(
            metrics
        )

        reasons = []

        if not row.get(
            "readable",
            False
        ):
            reasons.append(
                "unreadable"
            )

        else:

            if row.get(
                "low_information_candidate",
                False
            ):
                reasons.append(
                    "low_information"
                )

            if row.get(
                "dual_panel_candidate_v2",
                False
            ):
                reasons.append(
                    "dual_panel"
                )

            if not row.get(
                "standard_resolution",
                True
            ):
                reasons.append(
                    "nonstandard_resolution"
                )

        row[
            "needs_review_v2"
        ] = (
            len(reasons) > 0
        )

        row[
            "review_reason_v2"
        ] = "|".join(
            reasons
        )

        rows.append(
            row
        )


# ============================================================
# GUARDAR
# ============================================================

df = pd.DataFrame(
    rows
)

df = df.sort_values(
    [
        "patient_id",
        "relative_path"
    ]
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


# ============================================================
# RESUMEN
# ============================================================

print(
    "\n========================================"
)

print(
    "BEHSOF QUALITY AUDIT V2"
)

print(
    "========================================"
)

print(
    "\nPacientes:",
    df["patient_id"].nunique()
)

print(
    "Imágenes:",
    len(df)
)

print(
    "\nImágenes por centro:"
)

print(
    df["center"]
    .value_counts()
)

print(
    "\nLow-information candidates:"
)

print(
    df[
        "low_information_candidate"
    ].sum()
)

print(
    "\nDual-panel candidates:"
)

print(
    df[
        "dual_panel_candidate_v2"
    ].sum()
)

print(
    "\nNon-standard resolution:"
)

print(
    (
        ~df[
            "standard_resolution"
        ]
    ).sum()
)

print(
    "\nNeeds review:"
)

print(
    df[
        "needs_review_v2"
    ].value_counts()
)

print(
    "\nReview reasons:"
)

print(
    df.loc[
        df[
            "needs_review_v2"
        ],
        "review_reason_v2"
    ]
    .value_counts()
)

print(
    "\nArchivo generado:"
)

print(
    OUTPUT
)


# ============================================================
# VALIDACIÓN CON CASOS CONOCIDOS
# ============================================================

KNOWN_DUAL = [
    ("TAL18131", "Image.jpg"),
    ("TAL16001", "Image.jpg"),
    ("TAL09011", "Image.jpg")
]

KNOWN_LOW_INFO = [
    ("TAL06221", "Image4.jpg")
]


print(
    "\n========================================"
)

print(
    "VALIDACIÓN MANUAL"
)

print(
    "========================================"
)


for patient_id, filename in KNOWN_DUAL:

    match = df[
        (df["patient_id"] == patient_id)
        &
        (
            df["filename"]
            .str.lower()
            ==
            filename.lower()
        )
    ]

    print(
        f"\nDUAL conocido: "
        f"{patient_id} / {filename}"
    )

    if len(match) == 0:

        print(
            "  NO ENCONTRADO"
        )

    else:

        r = match.iloc[0]

        print(
            "  dual_panel_candidate_v2:",
            r[
                "dual_panel_candidate_v2"
            ]
        )

        print(
            "  left_content:",
            round(
                r[
                    "left_content_ratio"
                ],
                3
            )
        )

        print(
            "  center_content:",
            round(
                r[
                    "center_content_ratio"
                ],
                3
            )
        )

        print(
            "  right_content:",
            round(
                r[
                    "right_content_ratio"
                ],
                3
            )
        )

        print(
            "  center_drop_ratio:",
            round(
                r[
                    "center_drop_ratio"
                ],
                3
            )
        )


for patient_id, filename in KNOWN_LOW_INFO:

    match = df[
        (df["patient_id"] == patient_id)
        &
        (
            df["filename"]
            .str.lower()
            ==
            filename.lower()
        )
    ]

    print(
        f"\nLOW INFO conocido: "
        f"{patient_id} / {filename}"
    )

    if len(match) == 0:

        print(
            "  NO ENCONTRADO"
        )

    else:

        r = match.iloc[0]

        print(
            "  low_information_candidate:",
            r[
                "low_information_candidate"
            ]
        )

        print(
            "  roi_mean:",
            round(
                r[
                    "roi_mean_intensity"
                ],
                3
            )
        )

        print(
            "  roi_entropy:",
            round(
                r[
                    "roi_entropy"
                ],
                3
            )
        )

        print(
            "  roi_ratio_gt30:",
            round(
                r[
                    "roi_ratio_gt30"
                ],
                3
            )
        )

        print(
            "  edge_ratio:",
            round(
                r[
                    "roi_edge_ratio"
                ],
                3
            )
        )