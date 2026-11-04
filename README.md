# Fuga de información en la clasificación ecográfica de esteatosis hepática

Repositorio experimental del artículo provisionalmente titulado:

> **Fuga de información en la clasificación ecográfica de esteatosis hepática: comparación entre particiones por imagen y por paciente**

El estudio evalúa en qué medida una partición por imagen puede producir una estimación optimista del rendimiento cuando imágenes de un mismo paciente aparecen tanto en entrenamiento como en validación. Para ello se compara con una partición estricta por paciente bajo condiciones experimentales equivalentes.

Este trabajo deriva de una tesis sobre inteligencia artificial explicable aplicada a la esteatosis hepática, pero tiene una pregunta metodológica, un diseño y un alcance propios. El objetivo del artículo no es desarrollar el sistema completo de la tesis ni comparar arquitecturas.

## Pregunta de investigación

¿En qué medida la partición por imagen, al permitir que imágenes de un mismo paciente aparezcan en entrenamiento y validación, afecta el rendimiento estimado de un modelo de aprendizaje profundo para clasificar esteatosis hepática en ecografías, en comparación con una partición por paciente?

## Objetivo general

Evaluar el efecto de la partición por imagen, en comparación con la partición por paciente, sobre el rendimiento estimado de un modelo de aprendizaje profundo para la clasificación binaria de esteatosis hepática en ecografías del subconjunto BEH de BEHSOF, bajo condiciones experimentales controladas.

## Diseño resumido

- **Datos principales:** subconjunto BEH de BEHSOF.
- **Cohorte analítica:** 93 pacientes y 1,371 imágenes únicas, derivadas de
  una selección fuente de 95 pacientes y 1,388 imágenes elegibles.
- **Variable objetivo:** `Fibroscan S`, agrupada como `S0-S1 = 0` y `S2-S3 = 1`.
- **Arquitecturas:** EfficientNet-B0 como análisis principal y ResNet50 como
  comprobación de robustez, ambas mediante transferencia de aprendizaje y sin
  seleccionar una arquitectura ganadora.
- **Protocolos comparados:** cinco repeticiones de validación cruzada de cinco
  folds por imagen y por paciente.
- **Control experimental:** dentro de cada arquitectura se mantienen los mismos
  datos, preprocesamiento, aumentos, hiperparámetros y semillas emparejadas en
  ambos protocolos.
- **Unidad de evaluación:** paciente; las probabilidades fuera de muestra de sus imágenes se promedian antes de calcular las métricas.
- **Métrica principal:** AUROC a nivel paciente.
- **Comparación principal:** media de las cinco diferencias emparejadas
  `ΔAUROC = AUROC por imagen - AUROC por paciente` obtenidas con
  EfficientNet-B0; ResNet50 se informa separadamente como robustez.
- **Incertidumbre:** intervalo de confianza del 95 % mediante bootstrap pareado y estratificado a nivel paciente con 10,000 remuestreos.
- **Métricas secundarias:** balanced accuracy, sensibilidad, especificidad y matriz de confusión con umbral fijo de `0.5`.

La partición por imagen constituye deliberadamente la condición susceptible a fuga de información. La partición por paciente actúa como estrategia de referencia y debe mantener en cero el solapamiento de pacientes entre entrenamiento y validación.

## Alcance

El experimento principal se limita al centro BEH. TAL y el conjunto de Byra no forman parte del análisis principal. Grad-CAM, la comparación extensa de arquitecturas y la evaluación de aceptación clínica pertenecen al alcance de la tesis y no son objetivos de este artículo.

Algunos scripts y metadatos de TAL o Byra se conservan porque proceden de la copia de trabajo de la tesis y documentan la preparación previa de los datos. Su presencia en el repositorio no implica que se utilicen en el experimento principal.

## Estado actual

El proyecto se encuentra después de la construcción de la cohorte analítica,
las particiones experimentales y los pilotos exclusivamente técnicos, pero
antes de la congelación versionada y de los entrenamientos predictivos
definitivos. Por tanto, este repositorio todavía no contiene resultados de
rendimiento vigentes que deban interpretarse como hallazgos del artículo.

### Componentes completados y vigentes

- inventario, control de calidad técnico y revisión visual del subconjunto BEH;
- manifiesto maestro con trazabilidad de inclusiones y exclusiones previas;
- definición de la variable objetivo binaria S0–S1 frente a S2–S3;
- preprocesamiento en memoria, tamaño de entrada, canales, normalización y
  aumentos, incluida una auditoría visual aprobada en 40 de 40 imágenes;
- cohorte analítica corregida y auditada: 1,371 imágenes únicas de 93
  pacientes, derivadas de 1,388 imágenes fuente mediante 17 exclusiones
  trazables por duplicación exacta de contenido;
- cinco repeticiones reproducibles de validación cruzada de cinco folds bajo
  los protocolos por imagen y por paciente;
- auditoría de cobertura, exclusividad, semillas, solapamiento de pacientes,
  exposición a imágenes hermanas y coincidencias de hashes;
- implementación y pruebas automatizadas del entrenador vigente para
  EfficientNet-B0 y ResNet50; y
- implementación y pruebas sintéticas del evaluador preespecificado; y
- cuatro pilotos técnicos de una época que cubren la matriz completa de dos
  arquitecturas por dos mecanismos de partición, con lote 32 confirmado para
  ResNet50 y sin interpretar sus pérdidas o probabilidades como resultados
  científicos.

Los productos canónicos de estas etapas son
`metadata/BEH_analytic_cohort.csv`, los archivos situados en
`metadata/folds/` y los reportes de `metadata/qc/cohort_integrity/` y
`metadata/qc/partition_folds/`. Su generación y sus condiciones de aceptación
se describen en `docs/cohort_integrity.md` y `docs/partition_folds.md`.

### Componentes pendientes

- congelar y versionar el protocolo y el entorno auditado;
- realizar los entrenamientos definitivos, generar las predicciones fuera de
  muestra e incorporar los resultados al manuscrito.

El entrenador y evaluador vigentes consumen exclusivamente la cohorte y las
particiones canónicas. Los folds históricos ya fueron retirados del árbol
actual y permanecen recuperables en el historial de Git. El pipeline de datos
se aplica en memoria y no modifica ni reemplaza las imágenes originales.

## Estructura del repositorio

```text
tesis-xai-steatosis/
├── src/
│   ├── settings.py
│   ├── data/
│       ├── inventory_behsof.py
│       ├── audit_behsof_quality.py
│       ├── make_tal_contact_sheets.py
│       ├── make_beh_sample_contact_sheets.py
│       ├── make_beh_outlier_sheet.py
│       ├── build_behsof_manifest.py
│       ├── audit_beh_preprocessing.py
│       ├── audit_beh_input_pipeline.py
│       ├── build_beh_analytic_cohort.py
│       ├── build_beh_partition_folds.py
│       └── preprocess_behsof.py
│   ├── training/
│   │   └── train_beh_crossval.py
│   └── evaluation/
│       └── evaluate_partition_protocols.py
├── tests/
│   ├── test_build_beh_analytic_cohort.py
│   ├── test_build_beh_partition_folds.py
│   ├── test_evaluate_partition_protocols.py
│   ├── test_preprocess_behsof.py
│   └── test_train_beh_crossval.py
├── metadata/
│   ├── BEHSOF_manifest.csv
│   ├── BEH_analytic_cohort.csv
│   ├── folds/
│   │   ├── BEH_image_folds.csv
│   │   └── BEH_patient_folds.csv
│   └── qc/
│       ├── cohort_integrity/
│       ├── partition_folds/
│       └── preprocessing_audit/
│           ├── v1_review.csv
│           ├── v1_parameters.json
│           ├── v2_review.csv
│           └── v2_parameters.json
├── artifacts/
│   └── qc/
├── docs/
│   ├── cohort_integrity.md
│   ├── partition_folds.md
│   ├── evaluation_protocol.md
│   ├── preprocessing_audit.md
│   ├── input_pipeline.md
│   └── training_protocol.md
├── results/
├── config.example.yaml
├── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md
```

Los datos médicos originales se mantienen fuera del repositorio:

```text
tesis-xai-steatosis-data/
├── raw/
├── interim/
├── processed/
└── models/
```

La disposición local esperada es:

```text
Codigo_articulo/
├── tesis-xai-steatosis/
└── tesis-xai-steatosis-data/
```

## Configuración local

Se recomienda utilizar Python 3.13. El proyecto se ha verificado con Python 3.13.7 y el código requiere como mínimo Python 3.10.

Crear y activar un entorno virtual en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las versiones de PyTorch y Torchvision se fijan conjuntamente en
`requirements.txt`. Si se utilizará una GPU NVIDIA, debe instalarse la variante
de PyTorch compatible con el controlador y la plataforma CUDA del equipo,
conservando las mismas versiones declaradas del paquete.

Crear la configuración local:

```powershell
Copy-Item config.example.yaml config.local.yaml
```

`config.example.yaml` contiene una configuración reproducible de ejemplo. `config.local.yaml` contiene las rutas específicas de la máquina y está excluido mediante `.gitignore`.

Con la estructura de carpetas recomendada, la raíz de datos puede declararse así:

```yaml
paths:
  data_root: "../tesis-xai-steatosis-data"
```

## Pruebas automatizadas

Desde la raíz del repositorio y con el entorno virtual activado, la batería
completa se ejecuta mediante:

```powershell
python -m unittest discover -s tests -v
```

El protocolo congelado cuenta con 32 pruebas automatizadas distribuidas en
cinco áreas:

- 4 pruebas de construcción e integridad de la cohorte analítica;
- 6 pruebas de construcción, cobertura y solapamiento de los folds;
- 5 pruebas del preprocesamiento y las transformaciones de entrada;
- 8 pruebas del contrato de entrenamiento; y
- 9 pruebas del contrato de evaluación y del bootstrap.

Estas pruebas verifican reglas metodológicas, determinismo, dimensiones,
semillas, rutas, metadatos, unidades de análisis y rechazo de configuraciones
incompatibles. No miden el rendimiento predictivo ni sustituyen el
entrenamiento definitivo.

Además, la Fase 4 incluyó cuatro pilotos técnicos de una época, uno por cada
combinación de arquitectura y protocolo. Los pilotos comprobaron el flujo de
extremo a extremo y la factibilidad de memoria, pero sus pérdidas y
probabilidades no se consideran resultados científicos.

## Metadatos y control de calidad

`metadata/BEHSOF_manifest.csv` es la fuente de autoridad para determinar la muestra elegible. Cada fila representa una imagen original y conserva las decisiones de inclusión o exclusión mediante `include_final`, `qc_status` y `exclusion_reason`.

Para el experimento corregido del artículo, `metadata/BEH_analytic_cohort.csv`
es la capa derivada que aplica la auditoría de duplicados sobre las filas BEH
elegibles. Conserva las 1,388 filas fuente, sus SHA-256 y una decisión explícita
de inclusión analítica; no sobrescribe el manifiesto original ni modifica
`raw`. El procedimiento y sus condiciones de aceptación se documentan en
`docs/cohort_integrity.md`.

La selección inicial del manifiesto original utilizó:

```text
center == "BEH"
include_final == True
patient_eligible == True
```

Sobre esa selección, el experimento corregido utilizará exclusivamente las
filas de `metadata/BEH_analytic_cohort.csv` que cumplan
`analytic_include == True`. Las particiones canónicas de cinco repeticiones se
encuentran en `metadata/folds/` y su auditoría en
`metadata/qc/partition_folds/`; el procedimiento se documenta en
`docs/partition_folds.md`.

Los folds del protocolo invalidado fueron retirados del árbol de trabajo una
vez que el entrenador y el evaluador dejaron de depender de ellos. Su
procedencia permanece recuperable en el historial de Git y en el registro
metodológico; no constituyen entradas vigentes del experimento.

Los reportes situados en `metadata/qc/` son artefactos intermedios de trazabilidad. En particular, las alertas del reporte de outliers fueron generadas mediante heurísticas automáticas y posteriormente revisadas; no constituyen una clasificación vigente ni reemplazan al manifiesto final.

`metadata/qc/preprocessing_audit/` conserva las fichas revisadas y los parámetros de las candidatas v1 y v2. Las hojas de contacto asociadas permanecen localmente en `artifacts/qc/` y no se versionan.

La especificación y los comandos de verificación del pipeline están documentados en `docs/input_pipeline.md`. La evidencia tabular de su auditoría se guardará en `metadata/qc/input_pipeline_audit/`; las hojas de contacto permanecerán fuera de Git.

Los datasets originales y los contact sheets no se versionan. Los archivos de `raw` deben permanecer inalterados; todo preprocesamiento producirá copias derivadas en ubicaciones separadas.

## Política de versionado

La raíz del repositorio es `tesis-xai-steatosis`. La carpeta hermana `tesis-xai-steatosis-data` permanece fuera de Git y contiene los datos originales y derivados de gran tamaño.

Se versionan:

- código fuente y pruebas;
- documentación metodológica;
- configuración reproducible de ejemplo;
- manifiestos y metadatos tabulares necesarios para la trazabilidad;
- resultados agregados finales que no contengan imágenes médicas ni
  predicciones identificables por paciente.

Se excluyen mediante `.gitignore`:

- datos médicos originales y artefactos visuales derivados;
- historiales, predicciones y demás productos individuales de entrenamiento;
- `config.local.yaml` y archivos de variables de entorno;
- entornos virtuales, cachés y archivos temporales;
- pesos, checkpoints y modelos exportados.

Los cambios metodológicos se desarrollan en ramas específicas y se integran en `main` mediante Pull Requests después de revisar los archivos incluidos. Ningún script debe modificar los archivos de `raw`.

## Reproducibilidad

Los dos protocolos deberán ejecutarse con los mismos componentes experimentales, salvo la unidad utilizada para construir los folds. Las decisiones técnicas se fijarán antes de observar las métricas comparativas.

El código comprueba la identidad de la muestra y de los folds antes de entrenar,
impide que una ejecución final se aparte accidentalmente de la configuración
preespecificada y exige metadatos completos antes de aceptar predicciones para
la evaluación. Los resultados definitivos deberán generarse desde cero con el
árbol de trabajo revisado y versionado.

Los productos de cada ejecución se conservan localmente en
`results/training/` y están excluidos de Git. Después de verificar las 100
ejecuciones, las tablas agregadas de `results/evaluation/` podrán versionarse
como evidencia del artículo; los checkpoints permanecen fuera del repositorio.

## Privacidad y distribución de datos

Este repositorio no distribuye ecografías ni artefactos visuales derivados de los datasets médicos. Solo versiona código, configuración de ejemplo y metadatos necesarios para documentar el proceso experimental.

## Autor

Sebastián Díaz  
Ingeniería de Software  
Universidad Nacional Mayor de San Marcos
