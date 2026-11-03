# RabMV: máquina de potenciación consciente del registro de Valledupar

[![tests](https://github.com/jannerpareja/rabmv/actions/workflows/tests.yml/badge.svg)](https://github.com/jannerpareja/rabmv/actions/workflows/tests.yml)
[![docs](https://github.com/jannerpareja/rabmv/actions/workflows/docs.yml/badge.svg)](https://jannerpareja.github.io/rabmv/)
[![PyPI](https://img.shields.io/pypi/v/rabmv)](https://pypi.org/project/rabmv/)
[![Python](https://img.shields.io/pypi/pyversions/rabmv)](https://pypi.org/project/rabmv/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

*English version: [README.md](README.md).*

RabMV (*Registry-Aware Boosting Machine of Valledupar*) es una máquina de boosting explicable (un modelo
aditivo generalizado ajustado por boosting cíclico de Newton) que **conoce la estructura del registro** del
que vienen sus datos: qué conceptos mide cada ficha de vigilancia, en qué versión y de qué forma. Nació en
Valledupar (Colombia), donde los suicidios consumados y los intentos se registran en fichas distintas, y un
modelo ingenuo aprende la ficha en lugar de la persona. El método es general: lo puede usar cualquier
secretaría de salud cuyas fichas cambien con el tiempo o difieran entre instituciones.

Versión **0.1.0**. Validado en dos simulaciones congeladas y desarrollado con un registro real (Valledupar,
Colombia). Es un método de investigación, no una herramienta de decisión clínica.

- **Documentación** (en inglés): <https://jannerpareja.github.io/rabmv/> (fuentes en [docs/](docs/index.md)).
- **Licencia:** BSD-3-Clause. **Cómo citarlo:** el artículo del método está en preparación; mientras tanto,
  cite el software con [CITATION.cff](CITATION.cff) (GitHub lo muestra como «Cite this repository»).
- **Autores:** Janer Alfonso Pareja Gutiérrez (autor de correspondencia, aparejaj@uninorte.edu.co) y
  Margarita Rosa Gamarra Acosta, Universidad del Norte, Escuela de Ingeniería, Barranquilla, Colombia.

El código, los nombres y la documentación técnica están en inglés, como es convención en el software
científico. Esta guía en español explica cómo usarlo; los nombres entre `comillas de código` son los del
código.

## Financiación y agradecimientos

Con el apoyo del Ministerio de Ciencia, Tecnología e Innovación (Minciencias, Convocatoria 975 de 2026,
«Becas para el Cambio», Doctorado Nacional) y de la Universidad del Norte. Agradecemos a la Secretaría de Salud
Municipal de Valledupar, cuyo registro de vigilancia motivó el método y se usó en su desarrollo; esos datos no
se distribuyen y todos los archivos de datos de este repositorio son sintéticos. RabMV hace parte de la tesis
doctoral de J. A. Pareja Gutiérrez, dirigida por M. R. Gamarra Acosta.

## Qué hace

1. **Catálogo de conceptos** (`rabmv.schema.Catalog`, archivo TOML): qué se mide, con qué tipo y códigos.
2. **Mapeos por ficha y versión** (`rabmv.schema.Mapping`, TOML): cómo cada campo de cada ficha se traduce a
   un concepto, con su equivalencia (`exact`, `aggregated`, `partial`), qué significa un blanco (`blank`), el
   formato de respuesta (`response`) y las fechas de vigencia (`valid_from`, `valid_to`).
3. **Disponibilidad por caso** (`rabmv.harmonize`): medido (`measured`), sin dato (`missing`) o no medido
   por la ficha vigente (`not_measured`).
4. **Regla de elegibilidad** (`rabmv.evaluate_eligibility`): un concepto entra al modelo solo si se midió de
   forma comparable en ambas clases, en la resolución común más fina.
5. **Estimador** (`rabmv.RabMVClassifier`, interfaz de scikit-learn): paso de Newton óptimo por tramos
   (programación dinámica), bolsas vectorizadas y explicaciones aditivas exactas. Con `missing='marginalize'`
   un concepto que la ficha no midió aporta 0; con `categorical_features` las variables nominales no dependen
   de cómo se numeraron sus categorías. Con marginalización se admiten pares (`interactions`): cada tabla de
   pares se purifica (ANOVA funcional), de modo que un par con un concepto no medido aporta su promedio
   condicional exacto, 0. Sin estas opciones es idéntico bit a bit a EBMP v1.
6. **Verificaciones de auditoría** (`rabmv.run_checks`): detectan medición diferencial no declarada (A:
   constante en una clase; B: selección única; C: blancos registrados como «No»; E: un año sin el campo).

Los reportes contienen solo conteos agregados (de 1 a 4 suprimidos); ningún objeto guardado contiene filas.

## Instalación y pruebas

    python -m pip install rabmv

Desde una copia del repositorio, con las pruebas:

    python -m pip install -e ".[dev]"
    python -m pytest

Ejemplo en cinco pasos con un registro sintético público: `python examples/quickstart.py`.

Para la carpeta de desarrollo (`development/`): `python -m pip install -e ".[dev,development]"`.

## Ejemplo de mapeo (ficha 105, Cesar)

    form = "105_cesar"
    version = "2013-2022"
    valid_from = 2013-01-01

    [concepts.escolaridad]
    equivalence = "aggregated"
    resolution = "cinco_niveles"
    blank = "missing"
    recoding = { 0 = 0, 2 = 2, 3 = 3, 7 = 7, 9 = 9 }

Las claves del archivo van en inglés; los identificadores de los conceptos y los valores de cada ficha son los
de cada secretaría (pueden estar en español).

## Historial

- EBMP v1 (Competencia 1): copia congelada en `tests/reference/ebm_propio_v1.py` (huella 0396E530B408B66A).
- Fases 1 a 2e (oct. 2026): paquete `ebmp`, esquema consciente de las fichas, disponibilidad, elegibilidad,
  marginalización, categóricas y verificaciones.
- Fase 3 (26 oct. 2026): simulación ADEMP; método válido.
- Fase 4 (27 oct. 2026): primera auditoría con el esquema de Valledupar sobre datos reales (solo agregados).
- 27 oct. 2026: renombrado `ebmp` → `rabmv` (RabMV) y traducido al inglés (código, documentación y pruebas).
  Es solo un cambio de nombres: la prueba de regresión frente a v1 es idéntica bit a bit y la simulación da los
  mismos valores por semilla y las mismas decisiones (detalle en `simulation/FROZEN_PROTOCOL.json`).
- 0.1.0.dev10 (27 oct. 2026): interacciones purificadas con marginalización (`rabmv.core.purification`);
  diseño 2 del desarrollo (interacciones para RabMV, corrección por prior de los pesos, variante de
  diagnóstico de escolaridad).
- Fase 5 (28–31 oct. 2026): nivel de desarrollo con datos reales, repetición 1 (280 tareas, 0 errores).
- Fase 6 (31 oct.–1 nov. 2026): confirmación pre-especificada (`development/PROTOCOL_PHASE6.md`), validación
  temporal, modelos finales y análisis exploratorios del atajo (`development/PLAN_PHASE6_v2.md`).
- Simulación v2 (1 nov. 2026, `simulation_v2/`, plan en `simulation_v2/PLAN.md`): tratamiento de lo no
  medido en ambos motores (RabMV y EBM de InterpretML), transportabilidad a cinco registros destino y
  comparación de las verificaciones con la batería de Shukla et al. (2026); se congela antes de evaluar.
- 0.1.0 (3 nov. 2026): primera versión pública: refinamiento de la verificación B, pruebas matemáticas de la
  marginalización purificada, ejemplo sintético público, documentación, licencia, cita e integración continua.
