# Revisión de `pwilkin/strix-halo`

Fecha de captura: 2026-09-28.

## Identidad y alcance

- instalador/sitio: `pwilkin/strix-halo@f73872fe20dfdef460653f18e2fe63e5366ae958`;
- motor de referencia: `pwilkin/llama.cpp:strix-halo@b0f31f5876ef3856b55f5bb88072cc96e5effafe`;
- runtime experimental: `pwilkin/rocm-systems:ilintar-experiments@7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1`;
- licencia del instalador y el sitio: MIT; cada dependencia conserva su propia licencia.

La copia local se usa como material de investigación. No se han ejecutado los
instaladores ni se ha importado código. Las ideas siguientes son hipótesis para
una implementación propia desde contratos públicos y con pruebas independientes.

## Ideas que sí cambian el plan

1. **DeltaNet tiled primero.** Su mayor salto aislado proviene de mantener el
   estado recurrente en registros durante un tile de tokens y sustituir viajes
   por LDS por reducciones DPP/permlanex16 que preservan el orden. Será el primer
   prototipo de rendimiento del GDN, con CPU/oráculo y comparación bitwise o con
   tolerancia fijada antes del benchmark.
2. **Lector directo para PLE/n-gram.** Una tabla enorme y de acceso disperso no
   debe paginarse ciegamente por `mmap`. Se ensayará un lector `pread` asíncrono,
   con pool acotado, coalescing, caché de filas y control de residency. La cifra
   publicada 233,71→395,58 t/s (1,69×) es solo evidencia externa, no una meta.
3. **Ring de inputs para UMA.** El scheduler necesita buffers de entrada por
   generación/slot para que un replay de grafo no observe memoria reusada. Se
   formalizará ownership, fence y backpressure en lugar de depender de una
   etiqueta genérica de dispositivo integrado.
4. **PM4 retenido solo como experimento de decode.** El beneficio esperado está
   en grafos repetidos de decode; no en prefill con formas cambiantes. Requiere
   ABI/capabilities, invalidación correcta y fallback normal. No se tomará una
   rama experimental del runtime como requisito del producto.
5. **BF16 WMMA y fusiones por tráfico.** Evaluar dequant una vez por grafo,
   caché BF16, gate/up+SwiGLU, down routed y cadena residual BF16, pero con
   presupuesto explícito: el caché de pesos dequantizados compite por UMA.
6. **Corrección TOP_K.** Los empates forman parte del contrato. La referencia
   documenta que corregirlos costó rendimiento; nuestra puerta no permitirá
   optimizar suponiendo valores distintos.

## Resultados negativos que evitaremos repetir

- El camino QSA disperso completo quedó aproximadamente empatado con una
  atención densa WMMA incluso a 140k; indexer, compactación y top-k pueden
  consumir todo el ahorro. Denso exacto será el control obligatorio.
- Integrar el grafo sparse sin el kernel de atención seleccionado fue 1,71×
  peor: productor de layout, metadata y consumidor deben evaluarse juntos.
- El scoring WMMA del indexer fue neutral o peor y varios ajustes de tiles
  quedaron dentro del ruido. No entrarán por plausibilidad arquitectónica.
- Retained PM4 no explica las cifras de prefill publicadas porque los grafos no
  se repetían con propiedades estables.

## Protocolo que adoptamos

- un cambio conceptual por commit/experimento;
- baseline, candidato y control el mismo día y máquina;
- al menos tres repeticiones con dispersión, no mejor ejecución;
- guardia de memoria y exclusión de GPU para evitar OOM/contención;
- corpus de recall con hechos plantados y verificación de calidad;
- guardar también experimentos perdedores;
- comparar end-to-end además del microkernel.

Las cifras de pwilkin usan otro checkpoint, cuantización, runtime y protocolo.
Nunca se mezclarán directamente con Halogen, GUFO o nuestro futuro motor.
