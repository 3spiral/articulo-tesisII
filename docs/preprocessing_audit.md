# Auditoría del preprocesamiento BEH

## Propósito

Comprobar, antes del entrenamiento, que una única transformación determinística elimina información de interfaz sin recortar contenido ecográfico relevante. Esta auditoría no reabre las decisiones de inclusión y exclusión del manifiesto ni busca optimizar el rendimiento del modelo.

## Fuente de autoridad y muestra

La selección parte de `metadata/BEHSOF_manifest.csv` y admite únicamente registros que cumplan:

```text
center == "BEH"
include_final == True
patient_eligible == True
```

El procedimiento utiliza una muestra determinística de 40 pacientes distintos: 20 de cada clase y una imagen por paciente, con semilla 42. El balance se usa solo para cubrir ambas clases durante la inspección visual; no representa la prevalencia ni interviene en el entrenamiento.

## Primera transformación candidata (v1)

La regla todavía provisional es:

1. Recortar la imagen original de 1024 × 768 mediante la caja `(128, 0, 896, 768)`.
2. En el resultado de 768 × 768, enmascarar en negro las filas `[0, 48)` y `[720, 768)`.
3. Aplicar exactamente la misma regla a todas las imágenes, sin ajustes por paciente, clase o contenido.

El recorte horizontal produce geométricamente un cuadrado central. Las franjas superior e inferior requieren esta auditoría para confirmar que eliminan la interfaz y preservan el campo ecográfico; no se consideran óptimas por anticipado.

## Ejecución

Desde la raíz del repositorio:

```powershell
python -m src.data.audit_beh_preprocessing --candidate v1
```

El script solo lee los archivos originales. Para la primera candidata genera en `artifacts/qc/preprocessing_audit_v1/`:

- diez hojas de contacto con pares original/procesado;
- `preprocessing_audit_review.csv`, con una fila por imagen;
- `preprocessing_audit_parameters.json`, con la semilla, el hash del manifiesto y las coordenadas evaluadas.

La carpeta de salida está ignorada por Git porque contiene imágenes médicas derivadas. El script se detiene si la carpeta ya existe para no sobrescribir una revisión previa.

## Criterios de revisión

Para cada imagen se completarán en el CSV los campos:

- `peripheral_interface_removed`: `yes` si no queda texto, escala o interfaz periférica fuera del campo ecográfico; de lo contrario, `no`.
- `internal_annotation_present`: `yes` si permanecen calipers, líneas o textos de medición superpuestos dentro del campo ecográfico; de lo contrario, `no`. Este campo se registra por separado porque una máscara periférica global no puede eliminarlos sin afectar anatomía.
- `anatomy_preserved`: `yes` si el campo ecográfico relevante se mantiene; de lo contrario, `no`.
- `review_status`: `pass` únicamente cuando `peripheral_interface_removed` y `anatomy_preserved` sean `yes`; en otro caso, `fail`.
- `notes`: descripción breve de cualquier residuo o pérdida observada.

La regla candidata se confirmará solo si todas las imágenes revisadas obtienen `pass`. La presencia de anotaciones internas se resumirá y dará lugar a una decisión metodológica explícita antes del entrenamiento, pero no se confundirá con un fallo de la máscara periférica. Si alguna imagen falla, las coordenadas se ajustarán mediante una nueva regla global y se repetirá la auditoría en una salida separada, sin consultar métricas predictivas.

## Resultado

La revisión visual de las 40 imágenes se completó con los siguientes resultados:

- 38 imágenes con `review_status = pass`;
- 2 imágenes con `review_status = fail` por conservar un fragmento mínimo de texto periférico;
- 40 imágenes con `anatomy_preserved = yes`;
- 1 imagen con `internal_annotation_present = yes`.

Los casos que no superaron la regla candidata fueron:

| Índice | Paciente | Archivo | Motivo |
|---|---|---|---|
| 17 | `BEH89111` | `BEH89111/Image3.jpg` | Fragmento mínimo de texto periférico y presencia de una anotación interna. |
| 39 | `BEH74231` | `BEH74231/Image11.jpg` | Fragmento mínimo de texto periférico. |

La ficha completa se conserva localmente en `artifacts/qc/preprocessing_audit_v1/preprocessing_audit_review.csv`. Conforme al criterio preespecificado, la primera transformación candidata no se confirma.

## Segunda transformación candidata (v2)

Los dos fallos de v1 proceden de textos de medición situados en la esquina inferior izquierda del original. Después del recorte horizontal solo permanece su extremo dentro de las primeras columnas del cuadrado resultante, aproximadamente entre las filas 694 y 715.

Extender la máscara inferior a todo el ancho eliminaría una porción adicional del campo ecográfico. Por ello, v2 conserva todos los pasos de v1 y añade una máscara rectangular global sobre las columnas `[0, 64)` y las filas `[688, 720)` del resultado de 768 × 768.

La segunda candidata se genera mediante:

```powershell
python -m src.data.audit_beh_preprocessing --candidate v2
```

Sus artefactos se guardan en `artifacts/qc/preprocessing_audit_v2/`.

La revisión de v2 sobre la misma muestra obtuvo:

- 40 imágenes con `peripheral_interface_removed = yes`;
- 40 imágenes con `anatomy_preserved = yes`;
- 40 imágenes con `review_status = pass`;
- 1 imagen con `internal_annotation_present = yes`, correspondiente a los calipers del caso 17.

La máscara adicional eliminó los restos periféricos de los casos 17 y 39 sin producir pérdidas anatómicas observables en la muestra. Los calipers internos del caso 17 se conservan porque retirarlos mediante una regla global afectaría el campo ecográfico; su presencia queda documentada y el mismo tratamiento se aplicará en ambos protocolos experimentales.

Con estos resultados se confirma v2 como transformación definitiva de limpieza de interfaz BEH. Las coordenadas quedan congeladas antes de implementar el entrenamiento o consultar métricas predictivas.
