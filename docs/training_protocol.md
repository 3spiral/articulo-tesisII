# Protocolo de entrenamiento

El experimento compara dos esquemas de validación cruzada de cinco folds sobre
las mismas 1,388 imágenes BEH elegibles:

- `image`: asignación estratificada por imagen, con solapamiento deliberado de
  pacientes entre entrenamiento y validación;
- `patient`: asignación estratificada por paciente, sin solapamiento de
  pacientes.

Cada combinación de protocolo y fold constituye una ejecución independiente.
Se entrena EfficientNet-B0 inicializada con pesos de ImageNet, con todas sus
capas ajustables y una salida binaria de un logit. La configuración fija es:

- pérdida `BCEWithLogitsLoss` sin ponderación;
- optimizador AdamW;
- tasa de aprendizaje `1e-4` y decaimiento de pesos `1e-4`;
- lote de 32 imágenes;
- 20 épocas fijas;
- semilla 42;
- sin planificador de tasa de aprendizaje, detención temprana ni búsqueda de
  hiperparámetros;
- conservación exclusiva del checkpoint de la época final.

Un piloto técnico de una época puede utilizarse únicamente para verificar
memoria, tiempo y producción correcta de archivos. Sus pérdidas y probabilidades
no se usan para modificar la configuración ni forman parte de los resultados del
artículo. El piloto anterior y las ejecuciones parciales de la entrega
provisional fueron eliminados; cualquier comprobación nueva partirá desde cero.

## Ejecución del piloto

```powershell
python -m src.training.train_beh_crossval --protocol patient --fold 1 --epochs 1 --pilot
```

## Ejecución completa

```powershell
python -m src.training.train_beh_crossval --protocol both --fold all
```

Los historiales, metadatos y probabilidades finales se escriben en
`results/training/final/`. Los checkpoints se guardan fuera del repositorio, en
`tesis-xai-steatosis-data/models/article/final/`. Las predicciones corresponden
siempre al fold mantenido fuera del entrenamiento y se utilizarán después para
la evaluación por imagen y por paciente. La descarga reutilizable de los pesos
preentrenados queda también fuera de Git, en
`tesis-xai-steatosis-data/models/torch_cache/`.

El script bloquea una ejecución final si el número de épocas, el tamaño de lote,
la tasa de aprendizaje, el decaimiento de pesos o la semilla difieren de esta
configuración. Antes de entrenar también contrasta las rutas, pacientes,
etiquetas, folds, estrategias y semillas con el manifiesto y con los archivos de
partición versionados.

La evaluación exige los cinco folds completos de ambos protocolos. Cada archivo
de predicciones debe corresponder exactamente a las imágenes del fold declarado
y estar acompañado por metadatos que acrediten la configuración final, las
huellas del manifiesto y de la tabla de folds y el estado completo de la
ejecución. Los pilotos y ejecuciones parciales son rechazados.
