## 4.5 Modelos de perturbación (averías, tiempos de proceso estocásticos y llegadas dinámicas)

Esta sección formaliza los tres generadores de perturbación implementados en `envs/disruptions.py` sobre el entorno dinámico descrito en la Sección 4.2, desarrollando y justificando los valores concretos que la Parte D5 del marco metodológico dejó especificados solo hasta el nivel de familia de distribución y conjunto de niveles objetivo. Cada generador es una clase independiente con su propio flujo de números aleatorios (`numpy.random.Generator`), derivado del generador maestro mediante `numpy.random.default_rng(seed).spawn(3)` en el orden fijo (averías, tiempos de proceso, llegadas); esta construcción garantiza que activar o desactivar un eje de perturbación no perturba la secuencia de sorteos consumida por los otros dos ejes, propiedad indispensable para que las ablaciones del diseño experimental (Parte D4, Tabla de experimentos E4–E6) sean comparables entre sí bajo la misma semilla. La propiedad se verifica empíricamente, no solo por construcción, mediante `test_generators_independent_of_seed_spawn` (§4.5.6).

### 4.5.1 Averías de máquina (TTF/TTR)

Siguiendo la Parte D5 y la Sección B2.3, cada máquina $k$ alterna entre los estados *disponible* y *en reparación* según un proceso de renovación: el tiempo hasta el fallo (*time-to-failure*, TTF) se distribuye $\mathrm{Exp}(\lambda_k)$ y el tiempo de reparación (*time-to-repair*, TTR) se distribuye $\mathrm{Lognormal}(\mu_r,\sigma_r^2)$, con

$$\mathrm{MTBF}_k=\frac{1}{\lambda_k},\qquad \mathrm{MTTR}=\exp\!\Big(\mu_r+\frac{\sigma_r^2}{2}\Big),\qquad A_k=\frac{\mathrm{MTBF}_k}{\mathrm{MTBF}_k+\mathrm{MTTR}}.$$

**Calibración implementada.** En lugar de fijar $\lambda_k$ y $\mu_r$ directamente, el generador se parametriza por la disponibilidad objetivo $A_k$ y despeja los parámetros de las distribuciones subyacentes, siguiendo literalmente el procedimiento de dos pasos de D5 (`envs/disruptions.py`, clase `BreakdownGenerator`):

$$\bar p=\text{media de los tiempos de proceso de la instancia base},\qquad \mathrm{MTTR}=c_r\cdot\bar p,\qquad \sigma_r=\text{parámetro de forma configurado},$$
$$\mu_r=\ln(\mathrm{MTTR})-\frac{\sigma_r^2}{2}\quad\text{(de modo que }\mathbb{E}[\mathrm{Lognormal}(\mu_r,\sigma_r^2)]=\mathrm{MTTR}\text{ exactamente)},$$
$$\mathrm{MTBF}_k=\frac{A_k\cdot \mathrm{MTTR}}{1-A_k},\qquad \lambda_k=\frac{1}{\mathrm{MTBF}_k}.$$

donde $c_r$ (`mttr_multiplier`, por defecto $2.0$) fija el tiempo medio de reparación como un múltiplo del tiempo de proceso medio de la instancia. $A_k=1.0$ desactiva las averías de la máquina $k$ (no se programa ningún evento `BREAKDOWN`) en lugar de tratarse como una tasa nula.

**Justificación de los valores objetivo $A_k\in\{0.95,\,0.90,\,0.80\}$ (Parte D4, experimento E4).** Los tres niveles cubren un espectro operacionalmente interpretable de severidad de mantenimiento: $A_k=0.95$ representa un régimen de alta confiabilidad, propio de equipos con mantenimiento preventivo activo, típico de celdas CNC modernas; $A_k=0.90$ representa un régimen de degradación moderada; y $A_k=0.80$ es un régimen de estrés severo, empleado deliberadamente para forzar al agente a operar con una fracción sustancial de su capacidad indisponible y así exponer diferencias de robustez entre políticas que en el régimen de $0.95$ podrían resultar estadísticamente indistinguibles. Este mismo rango de severidad ($A_k$ en el orden de 0.8–0.95) es consistente con los mecanismos de recuperación ante averías reportados en los dos trabajos de la literatura revisada en el Capítulo 3 cuya mecánica de perturbación es más cercana a la de esta tesis (Lassoued et al. [38]; Zhang et al. [40]), lo que permite una comparación directa en el Experimento E4 sin necesidad de un ajuste compensatorio.

**Parámetros no fijados por D5 (asunciones señaladas explícitamente).** D5 especifica la familia de distribuciones y el conjunto de niveles $A_k$, pero no fija $c_r$ ni $\sigma_r$. Esta tesis adopta $c_r=2.0$ (un episodio de reparación dura, en promedio, el equivalente a dos operaciones típicas) como valor por defecto razonable — el propio ejemplo de D5 sugiere "MTTR ≈ 2·$\bar p$" — y $\sigma_r=0.5$ (coeficiente de variación $\mathrm{CV}=\sqrt{\exp(\sigma_r^2)-1}\approx 0.53$, variabilidad moderada) como una elección de forma sin respaldo empírico específico, y por tanto se documenta aquí como una asunción de modelado abierta a una prueba de sensibilidad posterior (véase el protocolo de validación, `../docs/planificacion/VandV_Protocol.md`, Fase 4).

**Sorteo del primer TTF y renovación.** Al inicio de cada episodio (`reset()`) se programa un evento `BREAKDOWN` inicial por máquina en $t=\mathrm{TTF}_k$; cada vez que una máquina se repara, se sortea de inmediato su siguiente TTF (proceso de renovación por máquina, no un único sorteo por episodio). Si la avería ocurre mientras la máquina procesa una operación, esta se preserva mediante la política de preempción descrita en la Sección 4.5.4.

### 4.5.2 Tiempos de proceso estocásticos

El tiempo de proceso efectivo de una operación se modela, exactamente como especifica D5, como el producto de la duración nominal por un multiplicador log-normal de media unitaria:

$$P_{j,h,k}=\bar p_{j,h,k}\cdot\xi,\qquad \xi\sim\mathrm{Lognormal}\!\Big(-\frac{s^2}{2},\,s^2\Big),\qquad \mathbb{E}[\xi]=1,\qquad \mathrm{Var}[\xi]=\big(e^{s^2}-1\big).$$

La elección de $\mu=-s^2/2$ es la que garantiza $\mathbb{E}[\xi]=1$ exactamente (de modo que el generador introduce variabilidad sin sesgar el makespan esperado al alza o a la baja respecto al caso determinista), forma cerrada verificada tanto analíticamente como empíricamente (§4.5.6). Los tres niveles $s\in\{0.10,\,0.25,\,0.50\}$ (Parte D4, experimento E5) corresponden a regímenes de variabilidad baja, media y alta, elegidos para trazar una curva de sensibilidad de la robustez frente a la magnitud de la incertidumbre de proceso, en lugar de un único punto de operación.

**Momento del sorteo (decisión de diseño señalada, D5 §10.5).** D5 fija la distribución pero no el instante en que se realiza el sorteo. Esta tesis sortea el multiplicador **una única vez, en el momento del despacho** de la operación, fijando `effective_duration` para el resto de la vida de esa operación — incluida una eventual pausa por avería bajo la política *preempt-resume* (Sección 4.5.4). Esto extiende de forma consistente el tratamiento que el entorno estático ya da a la duración (fija en el momento del despacho); la única diferencia es que el valor fijado es ahora una realización aleatoria y no una constante determinista.

### 4.5.3 Llegadas dinámicas de trabajos

Las llegadas de nuevos trabajos tras $t=0$ se modelan como un proceso de Poisson de tasa $\lambda_a$, con tiempos entre llegadas $\sim\mathrm{Exp}(\lambda_a)$, calibrado para alcanzar una utilización objetivo $\rho$ del taller:

$$\bar L=\text{carga total media por trabajo de la instancia base},\qquad m=\text{número de máquinas},\qquad \lambda_a=\frac{\rho\cdot m}{\bar L}\quad\Longleftrightarrow\quad \rho=\frac{\lambda_a\cdot \bar L}{m}.$$

**Justificación de los valores objetivo $\rho\in\{0.70,\,0.85,\,0.95\}$ (Parte D4, experimento E6).** Los tres niveles corresponden a los regímenes estándar de la teoría de colas para un sistema abierto: $\rho=0.70$ es un régimen de carga ligera con holgura de capacidad apreciable; $\rho=0.85$ representa una carga moderada-alta, comparable a la operación habitual de un taller bien planificado; y $\rho=0.95$ es un régimen de cuasi-saturación deliberadamente cercano al límite de estabilidad $\rho\to1$ (por encima del cual la cola de trabajos pendientes crecería sin cota en un sistema estacionario), utilizado como prueba de estrés de la capacidad del agente para mantener cronogramas factibles cuando casi no hay holgura de capacidad disponible.

**Seguridad de terminación (obligatoria, D5/§4.2.1 de la Sección 4.2).** Un proceso de Poisson puro nunca termina por sí mismo, lo que violaría la condición de "sistema terminante" requerida para el protocolo de réplicas de la Sección B9. Se imponen dos mecanismos independientes de corte: (a) un tiempo de corte `arrival_cutoff_time`, por defecto $0.8\cdot H_{\mathrm{nominal}}$ (dejando el último 20 % del horizonte nominal como periodo de "drenaje" sin llegadas nuevas), y (b) un tope duro `max_dynamic_jobs` sobre el número de trabajos generados. Ambos se verifican de forma independiente (`test_arrival_process_stops_at_cutoff`, `test_arrival_process_stops_at_max_dynamic_jobs`).

**Modelo de contenido de los trabajos entrantes (decisión de diseño señalada, no especificada por D5).** D5 especifica el proceso de llegada, no la composición de las operaciones del trabajo entrante. Cada trabajo nuevo conserva `ops_per_job` de la instancia base (preservando la rectangularidad; el enrutamiento flexible tipo FJSP queda fuera del alcance de este sprint), su orden de máquinas es una permutación aleatoria uniforme de las máquinas disponibles, y sus duraciones se generan por remuestreo bootstrap de la distribución empírica de duraciones de la instancia base (`instance.flat_durations`), en lugar de ajustar una distribución paramétrica adicional no exigida por D5. Esta elección se documenta explícitamente como una asunción de modelado, no como algo dictado por el marco original.

### 4.5.4 Política de preempción: *preempt-resume*

D1 deja explícitamente abierta la política de recuperación ante una avería que interrumpe una operación en curso ("según supuesto"). Esta tesis resuelve la ambigüedad adoptando **preempt-resume**: al producirse la avería, la operación se pausa (no se descarta), se calcula y conserva su tiempo remanente $\mathrm{remaining}=p_o-\mathrm{elapsed}$ bajo la duración efectiva ya sorteada (Sección 4.5.2), y se reanuda por exactamente ese tiempo remanente al completarse la reparación. La justificación, reutilizable directamente en la discusión metodológica, descansa en tres argumentos independientes:

1. **Realismo físico del dominio modelado.** El dominio de referencia (familia de instancias `ft`/`la`, mecanizado y ensamblaje discreto) no es un proceso por lotes químico donde una interrupción inutiliza el trabajo parcial; en mecanizado y ensamblaje, reanudar tras una reparación (cambio de herramienta, reajuste) es la práctica estándar, y descartar todo el progreso parcial en cada avería introduciría un sesgo pesimista sistemático frente al taller real que el modelo pretende representar.
2. **Consistencia con la literatura más cercana (Capítulo 3).** Los dos trabajos cuya mecánica de recuperación ante averías es más próxima a la de esta tesis (Lassoued et al. [38]; Zhang et al. [40]) modelan también la recuperación como reanudación del trabajo interrumpido, no como reinicio; adoptar *preempt-restart* aquí invalidaría la comparabilidad directa con esas líneas base en el Experimento E4.
3. **Trazabilidad no ambigua de la recompensa y de la cota inferior.** Bajo *resume*, `remaining_duration` es un escalar bien definido que se propaga por `estimated_makespan_dynamic` exactamente como la duración de una operación normal, evitando decisiones adicionales no motivadas (p. ej., si volver a sortear una duración "fresca" para el reintento bajo tiempos de proceso estocásticos, o cómo revisar retroactivamente el $LB(s)$ ya consumido por una operación descartada).

*Preempt-restart* se conserva como alternativa documentada y controlada por configuración (`preemption_policy="restart"`), reservada para una eventual prueba de sensibilidad solicitada por el comité, pero no forma parte de los criterios de aceptación de este sprint.

### 4.5.5 Resumen de asunciones de modelado no derivadas de D1/D5

La Tabla 4.5.1 recoge, para visibilidad del comité, las cinco decisiones de implementación que D1/D5 dejan genuinamente abiertas y que esta tesis resuelve de forma concreta y justificada, sin que ninguna de ellas contradiga lo explícitamente establecido en el marco original.

**Tabla 4.5.1.** Decisiones de diseño señaladas explícitamente (no dictadas por D1/D5).

| # | Decisión | Valor adoptado | Justificación |
|---|---|---|---|
| 1 | Forma del TTR ($\sigma_r$) | $\sigma_r=0.5$ (CV≈0.53) | Variabilidad moderada; sin fuente empírica específica — candidata a barrido de sensibilidad. |
| 2 | Contenido de trabajos entrantes | Remuestreo bootstrap de `flat_durations`; enrutamiento uniforme aleatorio | Evita imponer una distribución paramétrica adicional no exigida por D5. |
| 3 | Corte de llegadas / terminación | $0.8\cdot H_{\mathrm{nominal}}$ + tope `max_dynamic_jobs` | Garantiza un sistema terminante en el sentido DES (Sección B9). |
| 4 | Política de preempción | *preempt-resume* (por defecto) | Realismo del dominio + consistencia con [38],[40] + trazabilidad de la recompensa. |
| 5 | Momento del sorteo del multiplicador de duración | Una vez, al despachar la operación | Extiende el tratamiento de duración fija-al-despacho ya usado en el entorno estático. |

### 4.5.6 Validación estadística de los tres generadores

Antes de integrarse en el entorno dinámico completo, los tres generadores se validan de forma aislada, con $N=5000$ sorteos por aserción, semillas maestras fijas, y comparados contra el valor teórico calculado independientemente del propio objetivo reportado por el generador (de modo que un error de calibración en el generador también sería detectado). Los resultados de esta validación (`tests/test_disruptions.py`, 25/25 pruebas superadas) se resumen a continuación; el protocolo formal para la validación operacional del entorno completo, incluyendo el número de réplicas requerido y las pruebas de bondad de ajuste sobre corridas de producción, se especifica en el documento complementario `../docs/planificacion/VandV_Protocol.md`.

**Generador de averías** (`ft06`, $c_r=2.0$, $\sigma_r=0.5$):

| $A_k$ objetivo | MTBF objetivo | media empírica TTF | error | MTTR objetivo | media empírica TTR | error | $p$-valor KS (TTF vs. Exp) | disponibilidad empírica |
|---|---|---|---|---|---|---|---|---|
| 0.95 | 207.944 | 207.006 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9500 |
| 0.90 | 98.500 | 98.055 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.9001 |
| 0.80 | 43.778 | 43.580 | −0.45 % | 10.944 | 10.887 | −0.53 % | 0.929 | 0.8001 |

La prueba de Kolmogorov–Smirnov (`scipy.stats.kstest` contra $\mathrm{Exp}(\lambda_k)$) no rechaza la hipótesis exponencial ($p=0.929\gg0.01$), y la disponibilidad estacionaria del proceso de renovación recupera cada $A_k$ objetivo con un error inferior al 0.01 %.

**Muestreador de tiempos de proceso** (multiplicador log-normal de media 1):

| $s$ | media empírica | media objetivo | varianza empírica | varianza teórica $e^{s^2}-1$ | error |
|---|---|---|---|---|---|
| 0.10 | 0.9980 | 1.0 | 0.00991 | 0.01005 | −1.38 % |
| 0.25 | 0.9949 | 1.0 | 0.06339 | 0.06449 | −1.71 % |
| 0.50 | 0.9894 | 1.0 | 0.27667 | 0.28403 | −2.59 % |

**Proceso de llegadas** (Poisson calibrado a $\rho$, `ft06`):

| $\rho$ | $\lambda_a$ objetivo | $\lambda_a$ empírico | error |
|---|---|---|---|
| 0.70 | 0.12792 | 0.12691 | −0.79 % |
| 0.85 | 0.15533 | 0.15410 | −0.79 % |
| 0.95 | 0.17360 | 0.17223 | −0.79 % |

Estos tres resultados constituyen la evidencia de **verificación** (en el sentido de Sargent, 2011: el modelo se implementó correctamente respecto a su propia especificación matemática) de los generadores de perturbación. La **validación operacional** — que el entorno dinámico completo, ejecutado bajo estas calibraciones, reproduce un comportamiento admisible y estadísticamente consistente a la escala de los experimentos E0–E8 — es el objeto del protocolo formal descrito en `../docs/planificacion/VandV_Protocol.md`.
