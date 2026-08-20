# Protocolo de Verificación y Validación (V&V) del Entorno de Simulación

**Estado:** especificado ahora; ejecución empírica diferida a la fase de resultados (Capítulo 6, en conjunto con los experimentos E0–E8 de la Parte D4). Este documento fija el procedimiento antes de correr los experimentos, siguiendo la buena práctica de pre-registro ya adoptada en D7.

**Alcance:** el entorno de simulación de eventos discretos (`DynamicJSSPEnv` y sus tres generadores de perturbación, `envs/disruptions.py`) descrito en la Sección 4.2 y en la Sección 4.5. No cubre la validación del *desempeño* de la política aprendida (eso corresponde al protocolo estadístico de comparación de métodos de la Sección D6/5.3); cubre exclusivamente si el **simulador en sí** es una representación correcta y suficientemente fiel del sistema que pretende modelar.

---

## 1. Marco conceptual: verificación vs. validación (Sargent, 2011)

Siguiendo a Sargent (2011), se distinguen dos preguntas independientes que este protocolo responde por separado:

- **Verificación:** ¿el modelo computacional se implementó correctamente respecto a su propia especificación matemática (D5, D6)? Es una cuestión de depuración de software: trazas, pruebas unitarias, consistencia interna.
- **Validación:** ¿el modelo, ya verificado, representa con suficiente fidelidad el sistema real (o el sistema de referencia teórico) que pretende reproducir? Es una cuestión empírica sobre el comportamiento agregado del simulador, no sobre su código fuente.

La verificación de los tres generadores de perturbación (formas de las distribuciones, calibración de parámetros, independencia de flujos aleatorios) ya se completó y se reporta en `VALIDATION_WEEK6.md` (25/25 pruebas de `tests/test_disruptions.py`, resumidas también en la Sección 4.5.6). Este protocolo se ocupa de lo que ese informe **no** cubre: la validación operacional del entorno completo a la escala real de los experimentos, y el aparato estadístico (número de réplicas, pruebas de bondad de ajuste sobre corridas de producción, técnicas de validación de Sargent) necesario para ejecutarla con rigor cuando se corran los experimentos E0–E8.

---

## 2. Fase 1 — Determinación del número de réplicas

### 2.1 Fórmula

Para un sistema **terminante** (Sección B9; cada episodio de `DynamicJSSPEnv` termina por construcción, Sección 4.5.3), el número de réplicas independientes $R$ necesario para que el intervalo de confianza al $100(1-\alpha)\%$ de una métrica de salida (p. ej., $C_{\max}$, $RI$, $\bar T$) tenga un semi-ancho no mayor que $h$ se determina mediante:

$$R\ge\left(\frac{z_{1-\alpha/2}\cdot S}{h}\right)^2,$$

donde $S$ es la desviación estándar muestral de la métrica, estimada a partir de una **corrida piloto**, y $z_{1-\alpha/2}$ es el cuantil de la normal estándar correspondiente al nivel de confianza elegido (p. ej., $z_{0.975}=1.96$ para $\alpha=0.05$).

### 2.2 Procedimiento operativo

1. **Corrida piloto.** Ejecutar $n_0=10$ réplicas independientes (semillas distintas, mismo `DisruptionConfig`) por cada combinación de factores de la matriz experimental D4 (p. ej., cada celda de E4: una combinación de $A_k$ × regla de despacho / política). $n_0=10$ es el valor ya usado como referencia en el experimento E0 (validación DES) de la Parte D4.
2. **Estimar $S$** como la desviación estándar muestral de la métrica principal de esa celda sobre las $n_0$ réplicas piloto. Con $n_0<30$, usar el cuantil $t_{n_0-1,\,1-\alpha/2}$ de Student en lugar de $z_{1-\alpha/2}$ en la fórmula anterior (más conservador; converge a $z$ cuando $n_0$ crece).
3. **Definir $h$** como una fracción relativa de la media piloto $\bar X_{n_0}$ (recomendado: $h=0.05\cdot\bar X_{n_0}$, semi-ancho relativo del 5 %), salvo que el criterio de decisión de la hipótesis correspondiente (D7) exija una precisión distinta — p. ej., H4 exige distinguir un margen $\delta=2$ pp en RPD, lo que fija $h$ directamente.
4. **Calcular $R$** con la fórmula del §2.1. Si $R>n_0$, ejecutar $R-n_0$ réplicas adicionales con nuevas semillas y recalcular $S$ con la muestra ampliada (iterar una vez más si el nuevo $S$ desplaza $R$ de forma no trivial; en la práctica, dos iteraciones bastan).
5. **Verificación de suficiencia de las 30 réplicas nominales (E4/E5/E6).** La Parte D4 fija 30 réplicas como valor nominal para E4, E5 y E6. Este protocolo exige, para cada celda de esas matrices, reportar explícitamente si $R_{\text{calculado}}\le 30$ (30 réplicas es suficiente y el valor nominal se conserva) o si $R_{\text{calculado}}>30$ (el número de réplicas debe ampliarse para esa celda específica antes de reportar el resultado en el Capítulo 6) — no se debe asumir a priori que 30 es suficiente sin este chequeo.

### 2.3 Métricas sobre las que aplicar el criterio

Aplicar el procedimiento del §2.2 de forma independiente a cada una de las métricas principales de la Sección D6 que se reporten como promedio sobre réplicas: $\mathbb{E}[C_{\max}]$, índice de robustez $RI$, $\bar T$, y tiempo de decisión medio. Tomar $R=\max$ de los $R$ requeridos por cada métrica en esa celda, de modo que el intervalo de confianza declarado sea válido simultáneamente para todas ellas.

---

## 3. Fase 2 — Pruebas de bondad de ajuste sobre corridas de producción

La verificación de §1 confirma que los generadores producen, **de forma aislada**, muestras consistentes con sus distribuciones teóricas (Sección 4.5.6). La Fase 2 confirma que esa consistencia **se preserva** cuando los generadores operan embebidos dentro de episodios reales de `DynamicJSSPEnv` (donde el número de sorteos por episodio, el orden de consumo del flujo aleatorio y las interacciones entre ejes de perturbación son distintos del arnés de prueba aislado).

### 3.1 Procedimiento

Para cada experimento de la matriz D4 que involucre perturbaciones (E4, E5, E6), tras completar las $R$ réplicas de la Fase 1:

1. **Recolectar** todos los TTF, TTR realizados (E4), todos los multiplicadores de duración efectivamente aplicados (E5), y todos los intervalos entre llegadas efectivamente generados (E6) a lo largo de las $R$ réplicas — no solo del arnés de prueba unitario, sino de los eventos realmente disparados dentro de episodios completos.
2. **Ejecutar una única prueba de Kolmogorov–Smirnov** (`scipy.stats.kstest`) por distribución y por nivel de parámetro (es decir, una prueba por cada $A_k$, una por cada $s$, una por cada $\rho$), no una prueba por réplica: reservar el KS para una verificación de forma dedicada, siguiendo el mismo criterio de diseño ya aplicado en `tests/test_disruptions.py` (D5/§8 del `../docs/especificaciones/DynamicEnv_TechSpec.md`), dado que el KS es más costoso computacionalmente y su tasa de falsos positivos a $\alpha=0.05$ lo vuelve ruidoso si se aplica de forma masiva.
3. **Hipótesis:** $H_0$: la muestra realizada proviene de la distribución teórica calibrada (§4.5.1–4.5.3). Umbral de decisión $\alpha=0.01$ (más estricto que el 0.05 convencional, por la misma razón del punto anterior: reduce la tasa de rechazo espurio sin sacrificar potencia frente a una desviación real).
4. **Prueba complementaria para las llegadas:** además del KS sobre los tiempos entre llegadas, aplicar una prueba $\chi^2$ de bondad de ajuste sobre el conteo de llegadas por ventana de tiempo fija (contrastando contra $\mathrm{Poisson}(\lambda_a\cdot\Delta t)$), que valida la propiedad de conteo del proceso de Poisson de forma independiente de la propiedad de intervalos exponenciales ya cubierta por el KS.
5. **Reportar** el estadístico, el $p$-valor y la decisión para cada distribución/nivel en una tabla análoga a la de la Sección 4.5.6, explícitamente etiquetada como resultado de **validación operacional** (a escala de producción) y no de verificación aislada.

### 3.2 Criterio de aceptación

Si alguna prueba rechaza $H_0$ a $\alpha=0.01$: no descartar automáticamente el experimento; primero verificar que la calibración usada en la corrida de producción coincide exactamente con la calibración teórica (posible fuente de error: un parámetro de configuración mal propagado), y solo si la calibración es correcta, tratar el rechazo como evidencia de un problema real de implementación a resolver antes de reportar resultados del Capítulo 6.

---

## 4. Fase 3 — Validación estilo Sargent (lista de verificación aplicada a este entorno)

Checklist de las técnicas de validación de Sargent (2011) referenciadas en la Sección B9, con su instanciación concreta para este simulador DES+DRL.

### 4.1 Comparación con casos analíticos

| Caso analítico | Configuración del entorno | Comparación |
|---|---|---|
| Cola M/M/1 | Una única máquina, un único tipo de operación por trabajo, llegadas Poisson($\lambda_a$), tiempo de servicio Exp($\mu$) vía `processing_time_sigma` ajustado para aproximar exponencialidad, $\rho=\lambda_a/\mu<1$ | Contrastar el tiempo de espera medio simulado $\hat W$ y la longitud media de cola $\hat L_q$ contra las fórmulas cerradas $W=\frac{1}{\mu-\lambda_a}$, $L_q=\frac{\rho^2}{1-\rho}$ (ley de Little); tolerancia relativa 5 % con $R$ réplicas determinado por §2. |
| Renovación de disponibilidad (M/G/1 de reparación) | Una máquina, `BreakdownGenerator` con $A_k$ fijo, sin trabajos (o carga despreciable) | Contrastar la disponibilidad de largo plazo observada (fracción de tiempo *up*) contra $A_k$ objetivo — extensión a escala de episodio completo del chequeo ya hecho de forma aislada en §4.5.6. |
| Instancias con óptimo conocido | `ft06` ($C_{\max}^*=55$), `ft10` ($C_{\max}^*=930$), `ft20` ($C_{\max}^*=1165$), configuración sin perturbaciones (`DisruptionConfig()` por defecto) | El makespan bajo despacho óptimo (CP-SAT, cota superior) debe coincidir con el valor tabulado (OR-Library); ya cubierto indirectamente por `test_default_config_matches_static_env_exactly`, pero debe repetirse explícitamente contra el óptimo publicado, no solo contra el entorno estático, como parte de E0. |

### 4.2 Pruebas de degeneración

Verificar que, en los límites paramétricos degenerados, el simulador se reduce al comportamiento esperado y no exhibe artefactos:

- $A_k\to 1.0$: cero averías generadas; el entorno dinámico debe coincidir byte a byte con el entorno estático (ya verificado, `test_default_config_matches_static_env_exactly` y `test_zero_disruption_config_disables_every_axis`; repetir a escala de producción como parte de esta fase, no solo como prueba unitaria).
- $s\to 0.0$: duraciones deterministas; mismo chequeo de coincidencia con el entorno estático.
- $\rho\to 0$ / `target_utilization=None`: cero llegadas dinámicas; el número de operaciones del episodio debe permanecer igual al de la instancia base.
- $A_k\to 0^+$ (degradado a un valor extremo, p. ej. 0.5): la máquina debe pasar la mayor parte del tiempo en reparación; verificar que el makespan crece monótonamente a medida que $A_k$ disminuye, sin saltos ni instancias infactibles.
- $\rho\to$ próximo a 1 (0.95, ya en la matriz D4): verificar que el sistema sigue siendo terminante (los mecanismos de corte del §4.5.3 deben seguir garantizando finalización, incluso bajo carga cuasi-saturada) y que ningún episodio excede el límite superior de pasos (`test_episode_terminating_system_bounded_length`, repetido a la escala de E6).

### 4.3 Análisis de sensibilidad

- **Barrido de $\sigma_r$** (forma del TTR, señalado como asunción abierta en la Tabla 4.5.1, punto 1; ya sugerido como trabajo futuro en `VALIDATION_WEEK6.md`): correr E4 con $\sigma_r\in\{0.25,\,0.5,\,0.75\}$ manteniendo $A_k$ fijo, y verificar que la métrica de robustez responde de forma suave y en la dirección esperada (mayor $\sigma_r$ → mayor varianza de $C_{\max}$, sin necesariamente desplazar su media).
- **Barrido de $c_r$** (multiplicador de MTTR): análogo, para confirmar que el simulador no es frágil a la elección puntual $c_r=2.0$.
- **Ablación de política de preempción:** ejecutar la variante `preemption_policy="restart"` (ya implementada y controlada por configuración, no ejercitada por la suite de pruebas obligatoria) sobre el mismo conjunto de semillas que *resume*, y contrastar el desplazamiento de $\mathbb{E}[C_{\max}]$ y $RI$ — cuantifica el costo, en términos del objetivo, de la decisión de diseño justificada en la Sección 4.5.4.

### 4.4 Trazas y revisión por inspección

Extender el ejemplo de log de eventos ya transcrito en `VALIDATION_WEEK6.md` (§c) con al menos un episodio adicional por cada experimento de la matriz D4 que involucre perturbaciones combinadas, y someterlo a revisión manual (por el autor y, si es posible, por un revisor externo del comité) para verificación de plausibilidad operacional — no reemplaza las pruebas cuantitativas anteriores, mitiga el riesgo de que una prueba estadísticamente válida esconda un comportamiento sin sentido de dominio.

### 4.5 Validación de los datos de entrada

Confirmar que las estadísticas de la instancia base usadas para calibrar los tres generadores ($\bar p$, distribución empírica de duraciones, carga media por trabajo) son representativas del régimen de tamaño/dificultad que se pretende estudiar en cada experimento — repetir la calibración y el chequeo de §3 para al menos una instancia de cada familia usada en el diseño experimental (`ft`, `la`, `ta`, `MK`), no solo para `ft06`, antes de generalizar las conclusiones de robustez del Capítulo 6 a instancias de otra escala.

---

## 5. Fase 4 — Enlace con el protocolo estadístico de comparación (Sección D6/5.3)

Una vez validado el simulador (Fases 1–3), la comparación entre políticas/reglas de despacho sobre las réplicas generadas sigue el procedimiento ya especificado en D6: prueba de Wilcoxon de rangos con signo para comparaciones pareadas, prueba de Friedman con post-hoc de Nemenyi y diagrama de diferencia crítica para comparaciones de tres o más métodos, y reporte de tamaños de efecto (no solo $p$-valores). Este protocolo de V&V es un prerrequisito de ese procedimiento, no un sustituto: la validez de las conclusiones estadísticas de la Sección D6 depende de que el simulador que genera los datos haya superado las Fases 1–3 de este documento.

---

## 6. Cronograma de ejecución

Este protocolo se especifica ahora (Sprint de la Semana 6) pero **no se ejecuta esta semana**. Su ejecución está condicionada a la disponibilidad de la política entrenada (o, para las Fases 1–3, basta con el entorno y las líneas base PDR, sin requerir el modelo DRL) y se realiza en la fase empírica del Capítulo 6, en el siguiente orden:

1. Fase 1 (réplicas) y Fase 2 (bondad de ajuste) se ejecutan junto con cada experimento de la matriz D4 (E0 primero, como validación general del entorno; E4/E5/E6 después, específicos de cada eje de perturbación).
2. Fase 3 (Sargent) se ejecuta una vez, como parte de la validación general del entorno (Sección 6.1, "Validación del entorno de simulación"), antes de reportar cualquier resultado comparativo de desempeño.
3. Fase 4 se aplica en las Secciones 6.2–6.4 sobre los datos ya validados por las fases anteriores.

---

## 7. Resumen ejecutivo (checklist)

| Fase | Técnica | Artefacto/criterio | Cuándo |
|---|---|---|---|
| 1 | Cálculo de réplicas $R\ge(z\,S/h)^2$ | Piloto $n_0=10$ + iteración; $R$ reportado por celda experimental | Antes de cada bloque de E0–E8 |
| 2 | Kolmogorov–Smirnov (+ $\chi^2$ para conteos de llegadas) | Una prueba por distribución/nivel, $\alpha=0.01$, sobre datos de producción (no solo del arnés aislado) | Tras completar las réplicas de cada celda |
| 3a | Comparación analítica | M/M/1, renovación de disponibilidad, óptimos publicados (ft06/ft10/ft20) | Una vez, validación general del entorno |
| 3b | Pruebas de degeneración | $A_k\to1$, $s\to0$, $\rho\to0$, $A_k$ bajo, $\rho\to1$ | Una vez, validación general del entorno |
| 3c | Análisis de sensibilidad | Barrido $\sigma_r$, barrido $c_r$, ablación resume/restart | Una vez, validación general del entorno |
| 3d | Trazas / inspección | Logs de eventos ampliados, revisión manual | Una vez, validación general del entorno |
| 3e | Validación de datos de entrada | Repetir calibración/chequeo en ≥1 instancia por familia (`ft`,`la`,`ta`,`MK`) | Antes de generalizar conclusiones del Cap. 6 |
| 4 | Wilcoxon / Friedman+Nemenyi / tamaños de efecto | Sobre datos ya validados por 1–3 | Secciones 6.2–6.4 |
