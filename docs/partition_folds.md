# Particiones corregidas de BEH

## Estado

La Fase 2 está completada y auditada mediante I-19. Las particiones se
construyeron exclusivamente sobre la cohorte analítica corregida de 1,371
imágenes únicas correspondientes a 93 pacientes: 420 imágenes de 29 pacientes
S0-S1 y 951 imágenes de 64 pacientes S2-S3. Sus productos se regeneraron de
forma byte a byte determinística y conservan las huellas registradas en
`metadata/qc/partition_folds/parameters.json`.

Estas particiones definen las futuras predicciones fuera de muestra, pero no
son resultados predictivos ni contienen métricas de rendimiento. D-061 fue
ratificada explícitamente antes del entrenamiento definitivo; sus semillas no
se seleccionaron según pérdidas, probabilidades o AUROC.

## Diseño

La cohorte analítica confirmada se divide mediante cinco repeticiones completas.
Cada repetición utiliza cinco folds estratificados bajo dos protocolos:

- **por imagen:** cada ecografía se asigna individualmente;
- **por paciente:** todas las ecografías de una persona permanecen en el mismo
  fold.

Las raíces `42`, `43`, `44`, `45` y `46` alimentan una jerarquía reproducible
de `numpy.random.SeedSequence`. De cada raíz se derivan dominios separados para
la partición por imagen, la partición por paciente y el entrenamiento. Este
último genera una semilla por fold que se reutilizará en las condiciones
emparejadas.

## Justificación de la jerarquía de semillas

La jerarquía se fijó antes de los entrenamientos definitivos. Su propósito es
separar fuentes de aleatoriedad, conservar trazabilidad y organizar
comparaciones emparejadas; los números concretos no poseen una propiedad
estadística o clínica especial.

| Elemento | Justificación y alcance |
|---|---|
| Raíces `42` a `46` | `42` se heredó como convención reproducible del protocolo inicial y las cuatro raíces consecutivas amplían el diseño a cinco realizaciones completas. No se seleccionarán ni descartarán raíces según su rendimiento. |
| `SeedSequence(root).spawn(3)` | Crea dominios separados para partición por imagen, partición por paciente y entrenamiento. Así, modificar una finalidad no consume números de la corriente de otra ni cambia silenciosamente sus semillas. |
| Semillas distintas para ambos tipos de partición | Los protocolos asignan unidades diferentes y necesitan sorteos propios. El emparejamiento no exige folds idénticos, sino que ambas condiciones pertenezcan a la misma repetición planificada. |
| Cinco semillas hijas de entrenamiento | Cada fold recibe una corriente reproducible propia para inicialización, orden de lotes y aumentos. Esto evita reutilizar directamente el mismo entero raíz en todas las operaciones. |
| Reutilización por repetición y fold | La misma semilla de entrenamiento se asigna a las condiciones correspondientes de ambos protocolos y arquitecturas para organizar comparaciones emparejadas. No hace idénticos los pesos, lotes o aumentos cuando cambian los datos o la arquitectura. |
| Bootstrap con raíz independiente `20261030` | Separa el remuestreo estadístico de la generación de folds y del entrenamiento. Su valor es una convención Monte Carlo reproducible, no un parámetro predictivo. |
| Rutas canónicas separadas | `metadata/folds/` y `metadata/qc/partition_folds/` distinguen las cinco repeticiones corregidas de los productos históricos de una sola semilla. |

Las semillas derivadas, los dominios y las claves de generación se conservan en
`seed_schedule.csv`. La documentación de NumPy sobre `SeedSequence` respalda el
mecanismo de derivación; el número de raíces, su lista y el orden de los
dominios son decisiones de control preespecificadas, no valores demostrados
como óptimos.

## Ejecución

```powershell
python -m src.data.build_beh_partition_folds
```

La opción `--overwrite` solo debe emplearse para comprobar una regeneración
deliberada de esta misma etapa.

## Productos

- `metadata/folds/BEH_image_folds.csv`: 6,855 asignaciones, una por imagen y
  repetición.
- `metadata/folds/BEH_patient_folds.csv`: 465 asignaciones, una por paciente y
  repetición.
- `metadata/qc/partition_folds/seed_schedule.csv`: raíces, dominios, claves de
  derivación y semillas de entrenamiento.
- `metadata/qc/partition_folds/image_exposure.csv`: una fila por predicción OOF
  prevista —13,710 filas en total—, con `C_i`, `R_i`, exposición a imágenes
  hermanas e intersección de hash.
- `metadata/qc/partition_folds/fold_summary.csv`: resumen por protocolo,
  repetición y fold.
- `metadata/qc/partition_folds/parameters.json`: contrato, versiones y huellas
  de los productos.

## Condiciones de aceptación

- Cada imagen aparece exactamente una vez en validación por repetición y
  protocolo.
- Cada paciente aparece exactamente una vez en validación por repetición bajo
  el protocolo agrupado.
- La partición por paciente presenta cero identificadores compartidos y cero
  exposición a imágenes hermanas.
- Ningún protocolo comparte hashes exactos entre entrenamiento y validación.
- Las semillas de partición y entrenamiento se derivan de dominios separados.
- La regeneración de todas las salidas debe ser byte a byte determinística.

Las medidas del protocolo por imagen son descriptivas: no se utilizarán para
seleccionar repeticiones ni se interpretarán como una curva causal de
dosis-respuesta.
