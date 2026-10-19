# Partición por imagen y auditoría de solapamiento

El protocolo por imagen asigna individualmente las 1,388 ecografías elegibles
de BEH a cinco folds estratificados por `target_binary`. Antes de la asignación,
las filas se ordenan de forma estable por `relative_path`. Se utiliza
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

La asignación se conserva en `metadata/BEH_image_folds.csv` y no modifica el
manifiesto ni `metadata/BEH_patient_folds.csv`. Su auditoría se almacena en
`metadata/qc/image_level_folds/`:

- `fold_audit.csv` compara, fold por fold, tamaños, clases y solapamiento de
  pacientes en los protocolos por imagen y por paciente. La condición de
  referencia se carga desde `metadata/BEH_patient_folds.csv` y se contrasta con
  los pacientes, etiquetas y folds registrados en el manifiesto;
- `parameters.json` registra filtros, semilla, estrategia, versión de
  scikit-learn y huellas SHA-256 del manifiesto y de la partición por paciente.

Cada huella SHA-256 resume el contenido completo de un archivo para detectar
cambios accidentales. No es una huella individual de pacientes, no se calcula
sobre imágenes y no añade una nueva variable clínica o identificadora.

## Ejecución

```powershell
python -m src.data.build_beh_image_folds
```

El script se detiene si encuentra productos existentes. Para regenerarlos de
manera deliberada después de revisarlos:

```powershell
python -m src.data.build_beh_image_folds --overwrite
```

## Criterios de aceptación

- las 1,388 imágenes aparecen exactamente una vez;
- los folds contienen 277 o 278 imágenes;
- cada fold conserva aproximadamente la proporción de clases;
- en el protocolo por imagen, todos los pacientes de validación aparecen
  también en entrenamiento;
- en el protocolo por paciente, el solapamiento es cero.
