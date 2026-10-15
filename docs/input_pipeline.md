# Pipeline de entrada BEH

## Alcance

`src/data/preprocess_behsof.py` implementa el preprocesamiento y las
transformaciones confirmadas para el experimento principal con BEH. No crea un
dataset preprocesado permanente ni modifica los archivos originales.

## Secuencia común

1. cargar la imagen y corregir su orientación EXIF;
2. convertir explícitamente a RGB;
3. exigir una resolución original de 1024 × 768;
4. aplicar el recorte y las máscaras de la limpieza v2;
5. redimensionar directamente de 768 × 768 a 224 × 224 mediante interpolación
   bicúbica;
6. convertir a tensor `float32` y reescalar de `[0, 255]` a `[0, 1]`;
7. normalizar con media `[0.485, 0.456, 0.406]` y desviación estándar
   `[0.229, 0.224, 0.225]`.

## Diferencia entre entrenamiento y validación

Solo el pipeline de entrenamiento inserta, después del redimensionamiento y
antes de la conversión a tensor, una transformación afín con probabilidad 0.5:

- rotación entre −5° y +5°;
- traslación horizontal y vertical de hasta ±3 %;
- escala entre 0.95 y 1.05;
- sin cizallamiento;
- interpolación bilineal y relleno negro.

La validación es completamente determinística. Ambos protocolos de partición
deben utilizar estas mismas funciones; la unidad con la que se construyen los
folds será la única diferencia deliberada.

## Pruebas sin datos médicos

Las pruebas utilizan una imagen sintética y no acceden a `raw`:

```powershell
python -m unittest discover -s tests -v
```

Comprueban resolución y modo de la salida determinística, rechazo de una
resolución original inesperada, dimensiones y tipo del tensor, determinismo de
validación y reproducibilidad del entrenamiento al restablecer la semilla.

## Auditoría visual del aumento

La auditoría reutiliza exactamente las 40 rutas ya aprobadas en
`metadata/qc/preprocessing_audit/v2_review.csv`:

```powershell
python -m src.data.audit_beh_input_pipeline
```

Para que las 40 imágenes permitan inspeccionar la transformación, el script
fuerza su aplicación durante la auditoría. Esto no cambia la probabilidad 0.5
del pipeline real de entrenamiento. La secuencia aleatoria usa la semilla 42.

El script crea:

- hojas de contacto locales en `artifacts/qc/input_pipeline_audit/`;
- `metadata/qc/input_pipeline_audit/augmentation_review.csv`;
- `metadata/qc/input_pipeline_audit/augmentation_parameters.json`.

La ficha debe marcar `review_status = pass` únicamente cuando
`anatomy_preserved = yes` y `artifacts_acceptable = yes`. Se requieren 40 de
40 aprobaciones antes del entrenamiento. El script no sobrescribe resultados
existentes.

La revisión se completó el 15 de octubre de 2026. Las 40 imágenes conservaron
la anatomía y presentaron artefactos aceptables, por lo que las 40 obtuvieron
`review_status = pass`. La ficha y los parámetros resultantes se conservan en
`metadata/qc/input_pipeline_audit/`.
