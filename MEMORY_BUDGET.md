# Presupuesto UMA/GTT — borrador operativo

## Modelo correcto

En Strix Halo la RAM es físicamente compartida. GTT limita cuánto espacio puede
mapear `amdgpu`; no es memoria física adicional. Aumentarlo permite que la GPU
vea casi toda la RAM, pero no elimina la necesidad de reservar capacidad para
kernel, servicios, page cache, recuperación y picos temporales.

## Política propuesta para 128 GB

- BIOS: carve-out gráfico mínimo que permita el firmware (objetivo 0,5–2 GiB,
  nunca `Auto` sin comprobar el resultado).
- GTT: calcular desde la RAM real y ofrecer un perfil máximo cercano a 124 GiB.
  La cifra publicada por Halogen para un equipo nominal de 128 GB es
  `amdgpu.gttsize=126976` MiB y `ttm.pages_limit=32505856` páginas de 4 KiB.
- Admisión: conservar 16–20 GiB de margen efectivo para host, page cache,
  recuperación y picos, aunque el mapping limit sea mayor. El máximo mapeable
  no es el máximo asignable.
- Monitorizar `mem_info_gtt_used`, `MemAvailable`, memoria bloqueada/registrada,
  RSS/PSS, page cache y categorías internas del motor.
- Si un proceso muere sucio y quedan 35–45 GiB aparentes en GTT, impedir el
  reinicio automático del modelo, drenar el servicio y recomendar reinicio del
  host. La cifra es una observación publicada por Halogen y deberá reproducirse.

`amd_iommu=off` no será predeterminado: puede mejorar prefill en un host dedicado,
pero reduce aislamiento DMA y desactiva el NPU. Solo podrá existir como perfil
opt-in separado, con aviso y rollback. Tampoco se adoptarán sin medida
`amdgpu.vm_update_mode=0`, `amdgpu.noretry=0` o `amdgpu.sg_display=0`.

## Categorías que debe calcular el admission controller

1. pesos residentes y mappings;
2. PLE/n-gram residente, caché de filas y buffers de lectura directa;
3. KV por sesión;
4. estado recurrente GDN;
5. MTP, snapshots y rollback;
6. scratch, grafos, inputs ring y caché BF16/dequant;
7. runtime/librerías y margen del driver;
8. SO, servicios, page cache y reserva de recuperación.

El `config.json` oficial fijado declara 48 capas, una atención completa cada
cuatro capas, 2 cabezas KV y `head_dim=256`. Para un KV convencional BF16:

`12 capas × 2 cabezas × 256 × (K+V) × 2 bytes = 24 KiB/token/sesión`.

Por tanto, 262.144 tokens consumen 6 GiB por sesión y C4 consumiría 24 GiB,
antes de recurrencia, indexer, MTP, rollback y scratch. Es un límite inferior
de planificación, no una promesa de implementación; el primer contexto se
elegirá a partir de medidas C1/C2/C4.

La capa MTP de atención completa añade provisionalmente 2 KiB/token, dejando
26 KiB/token: 6,5 GiB a 262.144. Además, el estado next-step del drafter ya
tiene geometría fijada de `hc_count × hidden_size = 10.240` elementos por
sesión. El planner lo contabiliza como categoría separada: el escenario por
defecto usa 2 bytes/elemento (20.480 bytes/sesión), pero
`--mtp-draft-state-bytes` permite cambiar la hipótesis. Ese tamaño de elemento
es una **suposición de planificación**, no el dtype canónico del forward MTP. El indexer podría añadir ~768 B/token si la
compresión del config se traduce directamente a cache; debe confirmarse contra
el forward oficial. El estado GDN FP32 mínimo estimado es
`36 × 48 × 128 × 128 × 4 = 108 MiB/sesión`, más conv/history. Siete snapshots
completos serían ~756 MiB/sesión, de modo que se estudiará snapshot mínimo más
recompute. Estas cifras son inferencias de planificación, no contratos cerrados.

El planner publicará dos escenarios: PLE residente y PLE directa/no residente.
La caché de filas, page cache y cualquier caché BF16/dequant tienen tope duro y
contabilidad propia; se compararán `O_DIRECT` y/o `posix_fadvise(DONTNEED)` para
evitar que `pread` convierta la reserva del host en page cache invisible.

El tamaño en disco no equivale al working set. Como referencia externa,
Halogen publica unos 118 GiB de ficheros para base+overlay pero unos 68 GiB
residentes más una tabla n-gram mapeada; pwilkin mantiene fuera del resident set
una PLE de 27,5 GB mediante lectura directa. Ambos resultados requieren
reproducción en nuestro hardware y modelo elegido.

## UI administrativa

El panel podrá calcular y mostrar el perfil de arranque, comprobar RAM, BIOS,
kernel y parámetros actuales, y generar/aplicar un cambio versionado. Aplicar
se limita a GTT/TTM allowlisted y requiere arranque one-shot, confirmación,
health check y retorno automático a la entrada anterior si el nuevo arranque no
se confirma. IOMMU y parámetros experimentales solo se cambian por consola/SSH.
