# Protocolo de entrenamiento

## Estado

El entrenador vigente está implementado, cuenta con pruebas automatizadas y
completó los cuatro pilotos exclusivamente técnicos exigidos por D-063 para la
Fase 4. Todavía no se han ejecutado los entrenamientos definitivos. Las
pérdidas y probabilidades de los pilotos no constituyen resultados predictivos
vigentes ni se interpretarán en el artículo.

## Entradas canónicas

El entrenamiento utiliza exclusivamente:

- `metadata/BEH_analytic_cohort.csv`, filtrado por `analytic_include == True`;
- `metadata/folds/BEH_image_folds.csv`;
- `metadata/folds/BEH_patient_folds.csv`;
- `metadata/qc/partition_folds/seed_schedule.csv`; y
- `metadata/qc/partition_folds/parameters.json`.

Los archivos de folds situados directamente bajo `metadata/` pertenecen al
protocolo histórico y no son consumidos por el entrenador vigente.

## Diseño

Se ejecutarán 100 combinaciones:

`2 arquitecturas × 5 repeticiones × 2 protocolos × 5 folds`.

Las arquitecturas son EfficientNet-B0, como análisis principal, y ResNet50,
como robustez. Dentro de cada arquitectura se reutiliza la misma semilla de
entrenamiento para la combinación correspondiente de repetición y fold en
ambos protocolos. Las semillas proceden del calendario canónico y no pueden
sobrescribirse desde la línea de comandos.

## Configuración preespecificada y ratificada

| Componente | EfficientNet-B0 | ResNet50 |
|---|---|---|
| Pesos | `EfficientNet_B0_Weights.IMAGENET1K_V1` | `ResNet50_Weights.IMAGENET1K_V2` |
| Salida | Un logit | Un logit |
| Capas | Todas ajustables | Todas ajustables |
| Pérdida | `BCEWithLogitsLoss` sin ponderación | Igual |
| Optimizador | AdamW | AdamW |
| Tasa de aprendizaje | `1e-4` | `1e-4` |
| `weight_decay` | `1e-4` | `1e-4` |
| Épocas | 20 fijas | 20 fijas |
| Lote previsto | 32 | 32 |
| Precisión | `float32`, sin AMP | Igual |
| `num_workers` | 0 | 0 |
| Scheduler / detención temprana | No / no | No / no |
| Checkpoint | Última época | Última época |

Ambas arquitecturas reciben el mismo pipeline RGB de 224×224 ya auditado. No
se utilizan las transformaciones automáticas de inferencia asociadas a los
pesos, porque introducirían un procesamiento diferente del campo anatómico.

## Justificación de la configuración

La configuración se fijó antes del entrenamiento definitivo para estudiar el
efecto de la unidad de partición, no para optimizar o comparar clasificadores.
Ninguno de los valores siguientes se presenta como óptimo para BEH.

| Elección | Justificación y alcance |
|---|---|
| EfficientNet-B0 con `IMAGENET1K_V1` | Es el análisis principal elegido por su costo moderado y su entrada canónica de 224 × 224. El enum explícito evita que un alias cambie la inicialización y conserva la correspondencia con la normalización ya auditada. |
| ResNet50 con `IMAGENET1K_V2` | Es una familia arquitectónica distinta utilizada solo como robustez. V2 es una versión oficial y explícita de los pesos; no se usa `DEFAULT` para evitar cambios silenciosos entre versiones. No se comparará con V1 para elegir la que rinda mejor. |
| Transferencia y ajuste de todas las capas | Ambas redes parten de ImageNet y se adaptan por completo a la ecografía. Aplicar la misma política evita que congelar capas en una arquitectura introduzca una segunda diferencia deliberada. No se estudiarán estrategias alternativas de congelación. |
| Un logit y `BCEWithLogitsLoss` sin ponderación | Corresponde a la tarea binaria y mantiene una regla de optimización común. La ausencia de pesos evita reglas dependientes de la composición de cada fold; no implica que el desbalance sea irrelevante ni que esta pérdida sea óptima. |
| AdamW, `learning_rate=1e-4` y `weight_decay=1e-4` | Son valores conservadores y comunes heredados de la configuración preespecificada. Mantenerlos evita una búsqueda de hiperparámetros sobre los folds destinados a evaluación. |
| 20 épocas y checkpoint final | Fijan el mismo presupuesto de actualización y una regla de selección independiente del rendimiento observado. No se elige una época favorable ni se usa la validación para detención temprana. |
| Lote 32 | Es una convención común de factibilidad. El piloto confirmó que ambas arquitecturas lo admiten; ResNet50 reservó como máximo 3,530 MiB en la GPU disponible. No fue elegido por pérdidas o métricas. |
| `float32` sin AMP | Mantiene el régimen numérico sencillo e idéntico en ambas arquitecturas. La memoria disponible no exige precisión mixta, por lo que no se introduce esa fuente adicional de diferencias. |
| `num_workers=0` | Evita procesos secundarios de carga en el entorno Windows y simplifica el control reproducible de las transformaciones aleatorias. Se acepta menor paralelismo a cambio de estabilidad operativa. |
| Sin scheduler, detención temprana ni búsqueda de hiperparámetros | Reduce grados de libertad y evita utilizar los folds de evaluación para seleccionar configuraciones. El objetivo es comparar protocolos bajo un pipeline fijo, no maximizar cada arquitectura. |
| Pipeline común auditado | El mismo recorte, máscara, redimensionamiento, aumentos y normalización se aplican a ambas redes. No se adoptan las transformaciones automáticas asociadas a cada peso porque introducirían diferencias de campo anatómico entre arquitecturas. |

Estas razones se apoyan en las decisiones D-020, D-026 a D-029, D-032,
D-047, D-055 y D-062 del registro metodológico y en la documentación oficial
de Torchvision identificada allí como T-01 y T-07.

## Contingencia de memoria de ResNet50

El piloto comienza con lote 32. Solo si produce un error CUDA de memoria
reproducible en un entorno limpio se permite probar 16 y después 8. El primer
tamaño que complete el piloto se congelará para todos los entrenamientos
ResNet50 antes de ejecutar resultados definitivos.

La contingencia no puede activarse por pérdidas o métricas. No se introducirán
AMP, acumulación de gradientes ni activation checkpointing. Si el lote 8 falla,
el proceso se detendrá y exigirá una nueva decisión metodológica.

El piloto técnico de ResNet50 completó una época con lote 32 y reservó como
máximo 3,530 MiB de memoria CUDA en una NVIDIA GeForce RTX 4060 Laptop GPU de
8,188 MiB. Por tanto, no se activó la contingencia y el lote definitivo de
ResNet50 permanece en 32.

## Pilotos

Un piloto identifica exactamente una arquitectura, protocolo, repetición y
fold y ejecuta una sola época. Por ejemplo:

```powershell
python -m src.training.train_beh_crossval `
  --architecture resnet50 `
  --protocol patient_level `
  --repetition 1 `
  --fold 1 `
  --epochs 1 `
  --batch-size 32 `
  --pilot
```

Sus pérdidas y probabilidades no se utilizarán para modificar el protocolo ni
se informarán como resultados del artículo.

### Auditoría de la Fase 4

El 2026-11-03 se ejecutaron cuatro pilotos de una época, todos con la
semilla de entrenamiento `1587909317` correspondiente a la repetición 1 y el
fold 1:

- EfficientNet-B0 con partición por imagen y lote 32;
- EfficientNet-B0 con partición por paciente y lote 32;
- ResNet50 con partición por imagen y lote 32;
- ResNet50 con partición por paciente y lote 32;

D-063 fijó como criterio de cierre completar la matriz 2 × 2 antes de la Fase
5. Las cuatro combinaciones completaron carga, transformación, entrenamiento,
validación y escritura de historial, predicciones, metadatos y checkpoint. Las
predicciones cubrieron exactamente las rutas de validación esperadas,
permanecieron en el intervalo `[0, 1]` y sus huellas coincidieron con los
metadatos.

En ambos protocolos los lotes tuvieron forma `32 × 3 × 224 × 224`, tensores y
etiquetas `float32` y valores finitos. Reinicializar la misma semilla reprodujo
exactamente el orden, las etiquetas y los tensores del primer lote; la
transformación de validación produjo el mismo tensor al leer dos veces una
misma imagen. No se compararon ni utilizaron las pérdidas o probabilidades para
alterar el protocolo.

Entorno comprobado: Python 3.13.7, PyTorch 2.13.0+cu130, Torchvision
0.28.0+cu130, CUDA 13.0, cuDNN 9.20.0 y NVIDIA GeForce RTX 4060 Laptop GPU.
Los productos locales de piloto permanecen excluidos de Git mediante
`.gitignore`.

## Entorno de referencia para la ejecución definitiva

La auditoría previa al congelamiento del 2026-11-03 registró el siguiente
entorno de referencia:

| Componente | Versión o identificación |
|---|---|
| Sistema operativo | Windows 11, compilación 26200 |
| Python | 3.13.7 |
| PyTorch | 2.13.0+cu130 |
| Torchvision | 0.28.0+cu130 |
| Runtime CUDA de PyTorch | 13.0 |
| cuDNN informado por PyTorch | 9.20.0 (`92000`) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Memoria total de GPU | 8,188 MiB |
| Controlador NVIDIA | 596.21 |

Las dependencias directas están fijadas en `requirements.txt`. En esta misma
auditoría, las 32 pruebas automatizadas finalizaron correctamente y
`python -m pip check` no encontró dependencias rotas. Este registro describe
el entorno de referencia; no implica que las ejecuciones sean byte a byte
idénticas en hardware o plataformas diferentes.

## Entrenamiento definitivo

Una vez completadas las fases de piloto, congelación y versionado, la ejecución
completa utilizará los valores predeterminados:

```powershell
python -m src.training.train_beh_crossval
```

Cada combinación escribe historial, predicciones OOF, metadatos y checkpoint en
rutas que incluyen arquitectura, protocolo, repetición y fold. Los metadatos
registran las huellas de la cohorte, folds, calendario de semillas, parámetros,
predicciones, historial y checkpoint.

Los historiales y predicciones por ejecución se conservan localmente bajo
`results/training/` y no se versionan. Los checkpoints se almacenan en la
carpeta hermana de datos. Solo las salidas agregadas y verificadas de la etapa
de evaluación podrán incorporarse posteriormente al repositorio.

El entrenador rechaza configuraciones no preespecificadas, rutas o hashes
compartidos indebidamente, semillas incoherentes, productos parciales y
sobrescrituras accidentales.
