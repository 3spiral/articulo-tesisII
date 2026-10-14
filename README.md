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
- **Muestra elegible:** 95 pacientes y 1,388 imágenes incluidas.
- **Variable objetivo:** `Fibroscan S`, agrupada como `S0-S1 = 0` y `S2-S3 = 1`.
- **Arquitectura controlada:** EfficientNet-B0 con transferencia de aprendizaje.
- **Protocolos comparados:** validación cruzada de cinco folds por imagen y por paciente.
- **Control experimental:** mismos datos, preprocesamiento, aumentos, arquitectura, hiperparámetros y semillas en ambos protocolos.
- **Unidad de evaluación:** paciente; las probabilidades fuera de muestra de sus imágenes se promedian antes de calcular las métricas.
- **Métrica principal:** AUROC a nivel paciente.
- **Comparación principal:** `ΔAUROC = AUROC por imagen - AUROC por paciente`.
- **Incertidumbre:** intervalo de confianza del 95 % mediante bootstrap pareado y estratificado a nivel paciente con 10,000 remuestreos.
- **Métricas secundarias:** balanced accuracy, sensibilidad, especificidad y matriz de confusión con umbral fijo de `0.5`.

La partición por imagen constituye deliberadamente la condición susceptible a fuga de información. La partición por paciente actúa como estrategia de referencia y debe mantener en cero el solapamiento de pacientes entre entrenamiento y validación.

## Alcance

El experimento principal se limita al centro BEH. TAL y el conjunto de Byra no forman parte del análisis principal. Grad-CAM, la comparación extensa de arquitecturas y la evaluación de aceptación clínica pertenecen al alcance de la tesis y no son objetivos de este artículo.

Algunos scripts y metadatos de TAL o Byra se conservan porque proceden de la copia de trabajo de la tesis y documentan la preparación previa de los datos. Su presencia en el repositorio no implica que se utilicen en el experimento principal.

## Estado actual

Completado:

- inventario y auditoría técnica de BEHSOF;
- revisión visual de los casos seleccionados durante el control de calidad;
- manifiesto maestro con trazabilidad de inclusiones y exclusiones;
- definición de la variable objetivo binaria;
- folds reproducibles por paciente;
- selección del modelo controlado y del plan de evaluación.

Pendiente:

- confirmar mediante auditoría visual las coordenadas globales de limpieza de la interfaz BEH;
- cerrar tamaño de entrada, normalización y aumentos de datos;
- implementar la generación reproducible de folds por imagen;
- completar el preprocesamiento, entrenamiento, predicción fuera de muestra y análisis estadístico;
- incorporar los resultados del experimento.

No se han entrenado todavía los modelos del artículo. `src/data/preprocess_behsof.py` se mantiene como marcador de posición hasta confirmar el procedimiento definitivo de preprocesamiento.

## Estructura del repositorio

```text
tesis-xai-steatosis/
├── src/
│   ├── settings.py
│   └── data/
│       ├── inventory_behsof.py
│       ├── audit_behsof_quality.py
│       ├── make_tal_contact_sheets.py
│       ├── make_beh_sample_contact_sheets.py
│       ├── make_beh_outlier_sheet.py
│       ├── build_behsof_manifest.py
│       └── preprocess_behsof.py
├── metadata/
│   ├── BEHSOF_manifest.csv
│   ├── BEH_patient_folds.csv
│   └── qc/
├── artifacts/
│   └── qc/
├── docs/
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

Crear y activar un entorno virtual en Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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

## Metadatos y control de calidad

`metadata/BEHSOF_manifest.csv` es la fuente de autoridad para determinar la muestra elegible. Cada fila representa una imagen original y conserva las decisiones de inclusión o exclusión mediante `include_final`, `qc_status` y `exclusion_reason`.

Para el experimento principal solo se utilizarán filas que cumplan:

```text
center == "BEH"
include_final == True
patient_eligible == True
```

Los reportes situados en `metadata/qc/` son artefactos intermedios de trazabilidad. En particular, las alertas del reporte de outliers fueron generadas mediante heurísticas automáticas y posteriormente revisadas; no constituyen una clasificación vigente ni reemplazan al manifiesto final.

Los datasets originales y los contact sheets no se versionan. Los archivos de `raw` deben permanecer inalterados; todo preprocesamiento producirá copias derivadas en ubicaciones separadas.

## Reproducibilidad

Los dos protocolos deberán ejecutarse con los mismos componentes experimentales, salvo la unidad utilizada para construir los folds. Las decisiones técnicas se fijarán antes de observar las métricas comparativas.

Actualmente puede reproducirse la etapa de inventario y control de calidad. La ejecución integral del experimento se documentará cuando se implementen los módulos pendientes.

## Privacidad y distribución de datos

Este repositorio no distribuye ecografías ni artefactos visuales derivados de los datasets médicos. Solo versiona código, configuración de ejemplo y metadatos necesarios para documentar el proceso experimental.

## Autor

Sebastián Díaz  
Ingeniería de Software  
Universidad Nacional Mayor de San Marcos
