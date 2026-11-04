# Evaluación preespecificada

## Estado

El evaluador está implementado y probado con datos sintéticos, pero todavía
no se ha aplicado a resultados definitivos. Solo acepta las 100 ejecuciones
completas del protocolo vigente y rechaza pilotos, configuraciones distintas o
predicciones que no correspondan a los folds canónicos.

## Jerarquía analítica

Para cada arquitectura y repetición se reúnen primero las predicciones fuera de
muestra de los cinco folds.

1. **Análisis principal:** se promedian las probabilidades de las imágenes de
   cada paciente y se calcula un AUROC global sobre los 93 pacientes.
2. **Sensibilidad por imagen:** se calcula un AUROC global sobre las 1,371
   predicciones por imagen. Se etiqueta como análisis metodológico
   complementario, no como evidencia clínica independiente.
3. **Diagnósticos por fold:** los 25 AUROC de cada arquitectura y protocolo se
   conservan de forma descriptiva; no sustituyen el AUROC global.
4. **Métricas con umbral:** balanced accuracy, sensibilidad y especificidad se
   calculan por paciente con el umbral fijo 0.5. Las matrices de confusión se
   guardan por repetición y no se suman como observaciones independientes.

Para cada repetición se define
`delta_AUROC = AUROC_image_level - AUROC_patient_level`. El estimador por
arquitectura es la media de los cinco valores. EfficientNet-B0 constituye el
análisis principal y ResNet50 se informa separadamente como robustez.

El F1-score no forma parte del contrato de evaluación.

## Incertidumbre

El intervalo del 95 % utiliza 10,000 remuestreos pareados y estratificados por
paciente con la raíz independiente `20261030`. Cada remuestreo aplica las mismas
identidades y multiplicidades a ambos protocolos, las cinco repeticiones y las
dos arquitecturas. En la sensibilidad por imagen se incorporan todas las
imágenes del paciente seleccionado con la misma multiplicidad.

Este intervalo representa incertidumbre por remuestreo de pacientes
condicionada a los modelos ya entrenados y a las cinco realizaciones. No incluye
la variabilidad de volver a entrenar redes adicionales.

## Ejecución

Después de completar y verificar los 100 entrenamientos:

```powershell
python -m src.evaluation.evaluate_partition_protocols
```

## Productos

La salida se guarda en `results/evaluation/`:

- `patient_metrics.csv`;
- `image_sensitivity_metrics.csv`;
- `fold_diagnostics.csv`;
- `confusion_matrices.csv`;
- `delta_by_repetition.csv`;
- `metric_summary.csv`;
- `fold_diagnostics_summary.csv`;
- `delta_summary.csv`; y
- `parameters.json`, con el contrato y las huellas de entradas y salidas.

Los productos individuales de `results/training/` permanecen fuera del control
de versiones porque incluyen historiales y predicciones por imagen y paciente.
Después de verificar la integridad de las 100 ejecuciones, las tablas agregadas
de `results/evaluation/` podrán versionarse como evidencia reproducible del
artículo. Los productos solo se generarán después del entrenamiento definitivo.
