# Frozen pipeline samples

These examples are synthetic and do not represent the CODEFEST corpus.
The `passage: ` prefix is counted by the tokenizer but is not stored in chunks.

## SYN-ES-001

Detected language: `es` (ambiguous: `false`)

Preserved metadata:

```json
{"doc_id": "SYN-ES-001", "fuente": "synthetic/es/orbital.txt", "formato": "txt", "fenomeno": 1}
```

Original document:

El informe orbital registró un acercamiento a 3.14 kilómetros. El Dr. Ruiz confirmó la maniobra el 12 de mayo de 2026. ¿La alerta llegó a tiempo? Sí, quedó archivada.

Extracted sentences:

1. El informe orbital registró un acercamiento a 3.14 kilómetros.
2. El Dr. Ruiz confirmó la maniobra el 12 de mayo de 2026.
3. ¿La alerta llegó a tiempo?
4. Sí, quedó archivada.

### SYN-ES-001-chunk-0000 (47 tokens)

El informe orbital registró un acercamiento a 3.14 kilómetros. El Dr. Ruiz confirmó la maniobra el 12 de mayo de 2026. ¿La alerta llegó a tiempo? Sí, quedó archivada.

## SYN-EN-010

Detected language: `en` (ambiguous: `false`)

Preserved metadata:

```json
{"doc_id": "SYN-EN-010", "fuente": "synthetic/en/mixed.txt", "formato": "txt", "fenomeno": 20}
```

Original document:

The report is primarily English and documents a satellite event. Una nota breve permanece en español. However, the final analysis and conclusions remain in English. The fictional mission began with a review of optical measurements from three independent stations. Operators compared timestamps and confirmed that every clock used the same reference. Engineers inspected the antenna schedule before approving a second observation window. The science team recorded uncertainty values beside every calculated trajectory. A safety reviewer asked whether the proposed maneuver could be reversed. The flight group answered with a simulation and documented each assumption. Weather analysts then reported clear conditions above the coastal station. A second team verified the archive checksum and the source identifiers. The report distinguished direct observations from interpretations and forecasts. No automated system was allowed to make a final operational decision. The next section summarized battery reserves, thermal limits, and communication margins. Each table was converted into plain text for the synthetic corpus. Reviewers preserved dates, decimal values, quotations, and abbreviated titles. The final meeting compared the primary plan with two conservative alternatives. All participants signed the fictional record after resolving the open questions. The completed narrative remains intentionally long enough to exercise whole-sentence overlap across consecutive chunks.

Extracted sentences:

1. The report is primarily English and documents a satellite event.
2. Una nota breve permanece en español.
3. However, the final analysis and conclusions remain in English.
4. The fictional mission began with a review of optical measurements from three independent stations.
5. Operators compared timestamps and confirmed that every clock used the same reference.
6. Engineers inspected the antenna schedule before approving a second observation window.
7. The science team recorded uncertainty values beside every calculated trajectory.
8. A safety reviewer asked whether the proposed maneuver could be reversed.
9. The flight group answered with a simulation and documented each assumption.
10. Weather analysts then reported clear conditions above the coastal station.
11. A second team verified the archive checksum and the source identifiers.
12. The report distinguished direct observations from interpretations and forecasts.
13. No automated system was allowed to make a final operational decision.
14. The next section summarized battery reserves, thermal limits, and communication margins.
15. Each table was converted into plain text for the synthetic corpus.
16. Reviewers preserved dates, decimal values, quotations, and abbreviated titles.
17. The final meeting compared the primary plan with two conservative alternatives.
18. All participants signed the fictional record after resolving the open questions.
19. The completed narrative remains intentionally long enough to exercise whole-sentence overlap across consecutive chunks.

### SYN-EN-010-chunk-0000 (252 tokens)

The report is primarily English and documents a satellite event. Una nota breve permanece en español. However, the final analysis and conclusions remain in English. The fictional mission began with a review of optical measurements from three independent stations. Operators compared timestamps and confirmed that every clock used the same reference. Engineers inspected the antenna schedule before approving a second observation window. The science team recorded uncertainty values beside every calculated trajectory. A safety reviewer asked whether the proposed maneuver could be reversed. The flight group answered with a simulation and documented each assumption. Weather analysts then reported clear conditions above the coastal station. A second team verified the archive checksum and the source identifiers. The report distinguished direct observations from interpretations and forecasts. No automated system was allowed to make a final operational decision. The next section summarized battery reserves, thermal limits, and communication margins. Each table was converted into plain text for the synthetic corpus. Reviewers preserved dates, decimal values, quotations, and abbreviated titles.

### SYN-EN-010-chunk-0001 (76 tokens)

Reviewers preserved dates, decimal values, quotations, and abbreviated titles. The final meeting compared the primary plan with two conservative alternatives. All participants signed the fictional record after resolving the open questions. The completed narrative remains intentionally long enough to exercise whole-sentence overlap across consecutive chunks.

Overlap between the first two chunks:

- Reviewers preserved dates, decimal values, quotations, and abbreviated titles.

## SYN-ES-010

Detected language: `es` (ambiguous: `false`)

Preserved metadata:

```json
{"doc_id": "SYN-ES-010", "fuente": "synthetic/es/long.txt", "formato": "txt", "fenomeno": 10}
```

Original document:

Esta oración sintética deliberadamente extensa describe una secuencia imaginaria de observaciones orbitales, verificaciones humanas, cálculos científicos, registros temporales, comunicaciones entre estaciones, análisis de riesgos, decisiones documentadas y revisiones independientes que continúan con detalles adicionales para comprobar que una oración completa nunca se corta aunque llegue a superar el límite configurado por el tokenizer seleccionado para la primera validación del proyecto NERV y por ello debe conservarse íntegra como un único chunk sobredimensionado con toda su puntuación y significado original, mientras el equipo ficticio vuelve a revisar telemetría, catálogos, mapas, tablas, informes, mensajes, coordenadas, unidades, decimales, fechas, identificadores, firmas, sellos, permisos, copias, respaldos, alertas, umbrales, hipótesis, simulaciones, trayectorias, maniobras, ventanas de comunicación, reservas energéticas, tolerancias térmicas, velocidades relativas, estimaciones probabilísticas y criterios de aceptación, agregando deliberadamente más contexto continuo sobre estaciones imaginarias ubicadas en desiertos, islas, montañas y plataformas oceánicas, sobre operadores que comparan cada lectura con tres referencias independientes, sobre científicos que explican las incertidumbres a responsables de misión y sobre auditores que exigen conservar cada palabra de esta misma unidad sintáctica, además de observaciones complementarias acerca de sensores ópticos, radares, relojes atómicos, antenas orientables, baterías redundantes, enlaces cifrados, protocolos de emergencia, rutas alternativas, simulacros nocturnos, relevos de personal y decisiones reversibles, todo ello sin introducir un punto final anticipado para garantizar que el caso de prueba exceda de manera inequívoca doscientos cincuenta y seis tokens del encoder elegido y permita demostrar que la política congelada preserva la oración completa incluso cuando produce un chunk sobredimensionado.

Extracted sentences:

1. Esta oración sintética deliberadamente extensa describe una secuencia imaginaria de observaciones orbitales, verificaciones humanas, cálculos científicos, registros temporales, comunicaciones entre estaciones, análisis de riesgos, decisiones documentadas y revisiones independientes que continúan con detalles adicionales para comprobar que una oración completa nunca se corta aunque llegue a superar el límite configurado por el tokenizer seleccionado para la primera validación del proyecto NERV y por ello debe conservarse íntegra como un único chunk sobredimensionado con toda su puntuación y significado original, mientras el equipo ficticio vuelve a revisar telemetría, catálogos, mapas, tablas, informes, mensajes, coordenadas, unidades, decimales, fechas, identificadores, firmas, sellos, permisos, copias, respaldos, alertas, umbrales, hipótesis, simulaciones, trayectorias, maniobras, ventanas de comunicación, reservas energéticas, tolerancias térmicas, velocidades relativas, estimaciones probabilísticas y criterios de aceptación, agregando deliberadamente más contexto continuo sobre estaciones imaginarias ubicadas en desiertos, islas, montañas y plataformas oceánicas, sobre operadores que comparan cada lectura con tres referencias independientes, sobre científicos que explican las incertidumbres a responsables de misión y sobre auditores que exigen conservar cada palabra de esta misma unidad sintáctica, además de observaciones complementarias acerca de sensores ópticos, radares, relojes atómicos, antenas orientables, baterías redundantes, enlaces cifrados, protocolos de emergencia, rutas alternativas, simulacros nocturnos, relevos de personal y decisiones reversibles, todo ello sin introducir un punto final anticipado para garantizar que el caso de prueba exceda de manera inequívoca doscientos cincuenta y seis tokens del encoder elegido y permita demostrar que la política congelada preserva la oración completa incluso cuando produce un chunk sobredimensionado.

### SYN-ES-010-chunk-0000 (438 tokens)

Esta oración sintética deliberadamente extensa describe una secuencia imaginaria de observaciones orbitales, verificaciones humanas, cálculos científicos, registros temporales, comunicaciones entre estaciones, análisis de riesgos, decisiones documentadas y revisiones independientes que continúan con detalles adicionales para comprobar que una oración completa nunca se corta aunque llegue a superar el límite configurado por el tokenizer seleccionado para la primera validación del proyecto NERV y por ello debe conservarse íntegra como un único chunk sobredimensionado con toda su puntuación y significado original, mientras el equipo ficticio vuelve a revisar telemetría, catálogos, mapas, tablas, informes, mensajes, coordenadas, unidades, decimales, fechas, identificadores, firmas, sellos, permisos, copias, respaldos, alertas, umbrales, hipótesis, simulaciones, trayectorias, maniobras, ventanas de comunicación, reservas energéticas, tolerancias térmicas, velocidades relativas, estimaciones probabilísticas y criterios de aceptación, agregando deliberadamente más contexto continuo sobre estaciones imaginarias ubicadas en desiertos, islas, montañas y plataformas oceánicas, sobre operadores que comparan cada lectura con tres referencias independientes, sobre científicos que explican las incertidumbres a responsables de misión y sobre auditores que exigen conservar cada palabra de esta misma unidad sintáctica, además de observaciones complementarias acerca de sensores ópticos, radares, relojes atómicos, antenas orientables, baterías redundantes, enlaces cifrados, protocolos de emergencia, rutas alternativas, simulacros nocturnos, relevos de personal y decisiones reversibles, todo ello sin introducir un punto final anticipado para garantizar que el caso de prueba exceda de manera inequívoca doscientos cincuenta y seis tokens del encoder elegido y permita demostrar que la política congelada preserva la oración completa incluso cuando produce un chunk sobredimensionado.

## SYN-PT-010

Detected language: `es` (ambiguous: `false`)

Preserved metadata:

```json
{"doc_id": "SYN-PT-010", "fuente": "synthetic/pt/ambiguous.txt", "formato": "txt", "fenomeno": 30}
```

Original document:

Radar 2026. Sistema online. Status normal. Dados e data disponíveis para revisão.

Extracted sentences:

1. Radar 2026.
2. Sistema online.
3. Status normal.
4. Dados e data disponíveis para revisão.

### SYN-PT-010-chunk-0000 (22 tokens)

Radar 2026. Sistema online. Status normal. Dados e data disponíveis para revisão.
