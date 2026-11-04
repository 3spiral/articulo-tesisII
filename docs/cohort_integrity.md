# Cohorte analítica corregida de BEH

## Propósito

Esta etapa materializa una capa analítica derivada de las imágenes BEH que ya
estaban marcadas como elegibles en `metadata/BEHSOF_manifest.csv`. No modifica
el manifiesto original ni ningún archivo situado en `raw`.

La política aplicada es conservadora y fue fijada antes del nuevo experimento:

1. si un mismo SHA-256 aparece bajo identificadores de paciente diferentes, se
   excluyen de la cohorte analítica todos los registros de esas identidades;
2. si un SHA-256 se repite dentro del mismo paciente, se conserva la primera
   ruta según orden natural y se excluyen las restantes; y
3. cada inclusión y exclusión conserva su ruta, paciente, etiqueta, SHA-256,
   motivo y, cuando corresponde, la ruta retenida.

Para el hallazgo conocido, se excluyen `BEH42231` y `BEH43131` completos. En
`BEH62101` se conserva `Image9.jpg` y se excluye `Image10.jpg`.

## Ejecución

Desde la raíz del repositorio:

```powershell
python -m src.data.build_beh_analytic_cohort
```

El script se detiene si los productos ya existen. `--overwrite` solo debe
utilizarse después de comprobar que se pretende regenerar exactamente esta
etapa.

## Productos

- `metadata/BEH_analytic_cohort.csv`: las 1,388 filas fuente con SHA-256 y la
  decisión analítica explícita.
- `metadata/qc/cohort_integrity/exclusions.csv`: las 17 exclusiones.
- `metadata/qc/cohort_integrity/patient_image_distribution.csv`: número de
  imágenes de cada paciente incluido.
- `metadata/qc/cohort_integrity/summary.json`: huellas de entrada y salida,
  flujo de conteos, grupos duplicados y estadísticos de imágenes por paciente.

## Condiciones de aceptación

La ejecución solo escribe los productos si confirma:

- 1,388 imágenes y 95 pacientes en la fuente elegible;
- 93 pacientes y 1,371 imágenes incluidas analíticamente;
- 29 pacientes y 420 imágenes S0-S1;
- 64 pacientes y 951 imágenes S2-S3;
- una sola etiqueta por paciente;
- ningún SHA-256 repetido entre las imágenes incluidas;
- exclusión completa de `BEH42231` y `BEH43131`; y
- selección determinística de `BEH62101/Image9.jpg`.

Si alguna condición falla, la cohorte no debe utilizarse para construir folds.
