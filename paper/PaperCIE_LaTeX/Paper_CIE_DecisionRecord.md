# Decisión metodológica — Paper para Computers & Industrial Engineering

**Fecha:** 2026-08-21
**Estado:** manuscrito completo en inglés y español, formato Elsevier CAS (`cas-sc`), listo para revisión final del tutor antes del envío.

## El giro de enfoque

El tutor descartó los resultados anteriores. La reconstrucción no consistió en volver a correr experimentos, sino en **remedir el corpus preregistrado bajo una métrica corregida**. Nada del dato crudo cambió: las 4.500 ejecuciones perturbadas y las 43 instancias nominales son las mismas. Lo que cambió es cómo se califica el desempeño.

## La contribución central del paper

**Proposición 1.** Con `r` la desviación nominal respecto del óptimo certificado y `q` la realizada,

    RI = (q − r) / (1 + r),      ∂RI/∂r |_q = −(1+q)/(1+r)² < 0

El índice de robustez convencional es **estrictamente decreciente en la desviación nominal a desempeño realizado fijo**. Cuanto peor planifica un método, más robusto parece. Un `RI` negativo —reportado en la literatura como resiliencia excepcional— es la firma de un plan tan malo que la perturbación lo mejora.

Es una identidad, no una aproximación: verificada a 9,4×10⁻⁶ sobre las 150 celdas del corpus.

## Evidencia empírica

| Hallazgo | Valor | Fuerza |
|---|---|---|
| ρ(RI, desviación nominal) | −0,985 a −0,948 en los 6 regímenes de diez métodos, p ≤ 3×10⁻⁵ | fuerte |
| Orden por RI vs por calidad realizada | τ de Kendall negativo en 9/9; nunca coinciden | fuerte |
| Costo de seguir al índice | 15,8–49,3 pp (mediana 25,7) | fuerte |
| GNN-PPO bajo la métrica corregida | empata al mejor en 9/9, primer rango en 5/9 | moderado |
| Optimalidad nominal bajo perturbación | CP-SAT (r=0) realiza 42,0 % sobre el óptimo | descriptivo, n=4 |

## Correcciones aplicadas tras dos auditorías adversariales

1. **Referencia nominal.** El análisis promediaba 5 despliegues de la política aprendida mientras cada línea base aportaba uno. Corregido a semilla 0 determinista para todos, conforme al preregistro. Cambió `CompOracle` de −11,6 a −7,6 pp y la desviación nominal de RL de 29,1 % a 24,8 %.
2. **Llegadas dinámicas.** Se excluyen **5** reglas (spt, lpt, mwkr, edd, cr), no 2. Y la exclusión es **anticonservadora**, no conservadora: mwkr es la rival más fuerte de la política. Declarado explícitamente en §3.3 y §7.4.
3. **Nemenyi bajo llegadas.** La política **no** se separa del campo: empata con mor (brechas 0,11–0,20 frente a CD 0,79). El texto anterior afirmaba lo contrario.
4. **Valores p.** Los comparadores no alcanzan p<10⁻⁸; el máximo real es 4,03×10⁻⁵.
5. **Cola CVaR₉₀.** Promedia 3 de 30 realizaciones, no «el peor decil».

## Disciplina de reporte propuesta

1. Calificar contra una referencia **externa al método** (óptimo certificado).
2. Reportar la desviación nominal junto a cualquier cantidad de robustez.
3. Reportar cola, no solo media.
4. **Nombrar el comparador**: media de las nueve reglas (+233,4 pp), mejor por familia (+132,4 pp), u oráculo por instancia (−7,6 pp) dan veredictos opuestos sobre las mismas corridas.

## Pendientes antes del envío

- Verificar DOIs y rangos de página de `refs.bib` contra el registro editorial.
- Añadir ORCID en `\author[1]{...}[orcid=...]`.
- Depositar el artefacto (Zenodo o GitHub público) y apuntar allí la Declaración de disponibilidad de datos.
- Decidir el rol de la versión en español (C&IE publica en inglés).

## Reproducibilidad

`run_all.sh` regenera todo desde los CSV crudos: análisis, tablas, macros de prosa, figuras en ambos idiomas y los dos PDF. Ningún numeral del manuscrito está escrito a mano: todos provienen de `analysis/tex/macros.tex` o de las tablas generadas.
