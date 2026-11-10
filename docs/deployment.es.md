# Despliegue en otro hardware

[English](deployment.md) · [Interruptores individuales](toggles.md)

El despliegue se calcula a partir de medidas del dispositivo y de los recursos
del modelo, sin clases de hardware ni tablas basadas en la RAM instalada. Compile
para el destino GPU real: admitir un presupuesto de memoria no añade soporte de
kernels para otra arquitectura.

## Disponibilidad

El núcleo de despliegue ofrece sondeo, lectura TOML, presupuestos por rango e
inspección de DeepSeek. Las interfaces HTTP y las adaptaciones de motores descritas
aquí requieren una compilación que incluya esas integraciones; una compilación
solo del núcleo no incorpora todas las interfaces. Las pruebas de host cubren
los contratos. Siguen siendo necesarias la compilación nativa, la comprobación
de picos de memoria, la ejecución del servidor y la cualificación completa de
tokens, logits y estado en el hardware nuevo. Una inspección admitida no demuestra
esa cualificación.

| Motor | Opciones de despliegue implementadas | Límites actuales |
|---|---|---|
| Qwen | Codificación nativa; un nodo, dos réplicas, prefill explícito en pipeline de dos nodos | Ruta eager tiled registrada; un slot HTTP por réplica. Pipeline solo para prefill CLI y continuación en rango 0, no HTTP. YaRN y las rutas copy/graph se rechazan. |
| Nemotron | Q4_K_M; un nodo o réplicas independientes | Desactive borradores con `--draft off`. No están disponibles el dimensionamiento EXL3/con borradores ni la ejecución en par en esta adaptación. |
| DeepSeek | Codificación nativa; TP2 | Desactive la caché predeterminada con `--cache-profile off`, además de RAM de sesión, DSpark y etapas heterogéneas. Falta dimensionar workspace de trellis. Las estimaciones de prefill no admitidas se rechazan. |
| GLM | EXL3; TP2 | Solo el perfil default actual. Se rechazan caché/importación de prefijos, pipeline, intercambio externo y pilas truncadas. INT4 almacenado pendiente. |
| MiniMax | EXL3; TP2 | Desactive DSpark y caché/guardado/restauración de sesión; máximo 4096 filas. Ejecución INT4 pendiente. |

GLM y MiniMax pueden describir candidatos INT4, pero rechazan su ejecución no
disponible. La inspección INT4 de MiniMax puede conservar el cálculo de colocación
en un plan **rechazado**. Ni fijar INT4 ni `profile.allow_pending` habilita un
runtime ausente. El prefill desagregado forma parte del vocabulario del esquema,
pero estas adaptaciones no lo ejecutan. Los comandos sin despliegue mantienen sus
valores predeterminados y contratos de admisión propios.

## Inspección antes de cargar

Añada estas opciones al comando habitual del motor, conservando los recursos de
modelo/estado, el perfil y los argumentos de rango/transporte obligatorios:

```sh
--deploy auto --deploy-check
--deploy deployment.toml --deploy-check
```

`--deploy auto` busca entre las opciones del proveedor. `--deploy FILE` restringe
la búsqueda mediante TOML. `--deploy-check` imprime el plan y termina antes de
cargar los datos de pesos, construir los recursos del dispositivo o conectar el
par; los metadatos deben seguir siendo legibles. La inspección del par necesita
por tanto un inventario medido del otro nodo. Revise `refused`, `refusal`, los
motivos de los candidatos, `requested`, `chosen` y los presupuestos por rango.
No ejecute un plan rechazado. Quite `--deploy-check` para ejecutar una configuración
admitida, conservando los demás argumentos.

El sondeo observa el destino gfx, las CUs, GTT unificada o VRAM dedicada, los bytes
utilizables totales y los libres actuales. Los datos ausentes o ambiguos quedan
desconocidos y no justifican una admisión. Los nodos proceden del inventario medido;
el enlace procede del transporte de lanzamiento y del fichero, sin inventar un
nodo remoto ni medir automáticamente el ancho de banda. Los pares intercambian
medidas recientes en orden de rango y acuerdan el hash antes de cargar. Las réplicas
son procesos independientes gestionados por el orquestador; el despliegue no crea
procesos ni configura la red.

Para cada candidato y rango se reserva, por este orden:

1. KV y otro estado persistente para el contexto y los slots solicitados.
2. Workspace para el prefill/decode admitido.
3. Pesos, conversiones de formato, staging y otros recursos identificados.

El presupuesto es la memoria utilizable libre medida menos el margen de seguridad.
En memoria unificada siguen vigentes las comprobaciones nativas de host/GTT; la
RAM no es un segundo presupuesto que sumar a GTT. Todos los rangos deben caber.
Nunca se reducen el contexto ni los slots fijados. Los slots no fijados se buscan
desde el máximo del proveedor hacia abajo. El orden de topologías/formatos depende
del proveedor; no es un optimizador universal de rendimiento. Las comprobaciones
del asignador nativo siguen siendo determinantes.

## Contrato TOML

El esquema de entrada es `strix-deploy-v1`; el JSON de salida usa
`strix-deployment-v1`. Son restricciones, no conversiones de modelo ni ajustes
que establezcan cualificación numérica.

| Campo | Efecto y causas de rechazo |
|---|---|
| `schema` | Cadena exacta obligatoria. Se rechazan claves desconocidas, duplicados, tipos incorrectos y alias en conflicto. |
| `model` | Identificador opcional del proveedor; otro identificador se rechaza. Omítalo para usar el proveedor activo. No es una ruta de modelo. |
| `hardware.nodes`, `hardware.link` | Exigen el número de observaciones y un transporte compatible con el lanzamiento. Un nodo puede usar `auto`; varios nodos requieren enlace explícito. No se inventan datos remotos. |
| `topology.mode` | `auto`, `single`, `replica` (alias `replicas`), `tp2`, `pipeline`, `disaggregated`. Solo se ejecutan las opciones anunciadas por el proveedor. TP2 exige dos nodos medidos. |
| `topology.ranks` | Alias del número de nodos; debe coincidir si también se especifica ese número. No inicia procesos. |
| `memory.context` | Capacidad positiva en tokens; se aplica al contexto/capacidad del motor. Si KV no cabe, se rechaza en vez de recortar contexto. |
| `memory.slots` | Número positivo explícito que nunca se reduce. Omítalo para buscar, sujeto al adaptador y a las restricciones explícitas de CLI. |
| `memory.policy` | Solo está implementado `kv-first`; se rechaza `formats-first`. |
| `memory.margin_gib` | Reserva finita no negativa convertida a bytes. Si supera la memoria libre, no cabe. |
| `format.decode`, `format.prefill` | `auto` o un nombre exacto del proveedor. Se rechazan combinaciones no admitidas: Qwen/DeepSeek usan actualmente `native`, Nemotron `q4_k_m`, GLM/MiniMax `exl3`. |
| `format.resident_layers`, `format.total_layers`, `format.jit_slots` | Fijan la colocación exacta, incluido cero. Las capas residentes no pueden superar el total; la tupla completa debe coincidir con un candidato anunciado. No se inventan colocaciones parciales arbitrarias. |
| `profile.name`, `profile.allow_pending` | El perfil debe coincidir con el seleccionado en el lanzamiento. `allow_pending` solo acepta `false`; no cualifica una ruta provisional. |
| `devices` | Array opcional de medidas independientes por nodo, en orden de rango para pares. Datos ausentes, incoherentes o insuficientes impiden la inspección. La ejecución del par los sustituye por medidas recientes. |
| `pins` | Tabla opcional de cadenas que repite selecciones explícitas canónicas con valores idénticos. No permite sobrescribirlas; las diferencias se rechazan. Normalmente se omite. |

Se aceptan los alias raíz `nodes`, `link`, `context`, `slots`, y `topology`/`profile`
como cadenas. No combine un campo raíz con su equivalente anidado. Este esquema
no tiene claves `hardware.gfx`, `min_gtt_gib` ni `format.verify_min_rows`. Los datos
del dispositivo describen observaciones, no clases de hardware.

### Un fichero por motor

Estos contextos, slots y márgenes son ejemplos de política, no mediciones de
capacidad ni rendimiento. Cada fichero debe pasar inspección con sus recursos y
argumentos reales. Los ejemplos multinodo requieren además el inventario indicado
más abajo.

Qwen, un nodo (`qwen.toml`):

```toml
schema = "strix-deploy-v1"
model = "qwen3.8"
[hardware]
nodes = 1
link = "none"
[topology]
mode = "single"
[memory]
policy = "kv-first"
context = 8192
slots = 1
margin_gib = 8
[format]
decode = "native"
prefill = "native"
```

Réplicas de Nemotron (`nemotron.toml`):

```toml
schema = "strix-deploy-v1"
model = "nemotron-bolt-e4b"
[hardware]
nodes = 2
link = "tcp"
[topology]
mode = "replica"
[memory]
context = 8192
slots = 1
[format]
decode = "q4_k_m"
prefill = "q4_k_m"
```

Inicie cada réplica por separado con `--deploy-node 0` o `--deploy-node 1` y
`--draft off`; se vuelve a medir el dispositivo local en ese índice al ejecutar.
El campo de enlace no convierte las réplicas en un par tensor-paralelo.

Par DeepSeek (`deepseek.toml`; se infiere el identificador del modelo activo):

```toml
schema = "strix-deploy-v1"
[hardware]
nodes = 2
link = "usb4auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "native"
prefill = "native"
```

Par GLM (`glm.toml`):

```toml
schema = "strix-deploy-v1"
model = "glm-5.3"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

Par MiniMax (`minimax.toml`):

```toml
schema = "strix-deploy-v1"
model = "minimax-m3.1-flash"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

Use la escritura del transporte que acepte el motor seleccionado. Para TP2 mantenga
un proceso por rango y la configuración habitual de enlace/transporte. Para
inspeccionar sin conexión, añada una tabla `[[devices]]` por nodo. Incluya los seis
campos: `gfx_target` (cadena), `cu_count` (entero), `mem_total_bytes` y
`mem_free_bytes` (enteros), `unified` (booleano) y `source` (cadena no vacía que
describa la procedencia de la medida). Copie observaciones reales; no duplique el
dispositivo local como sustituto de otro nodo sin medir. Los bytes libres no
pueden superar el total. Mantenga privados los inventarios y recibos operativos.

## Formato según el régimen de ejecución

La regla prevista distingue reutilización de filas, no nombres de hardware. En
decode con `R = 1` suele dominar el tráfico de pesos: EXL3 compacto puede minimizar
los bytes leídos. En verificación ancha o trabajo por lotes, INT4 puede ganar si
`rows >= R*`: el umbral del motor/kernel a partir del cual reutilizar trabajo
matricial compensa el almacenamiento y la conversión adicionales. Prefill usa INT4
donde el runtime y el perfil están disponibles y cualificados. `R*` no es una
constante universal ni un campo TOML.

Una vez integrados y cualificados los runtimes INT4, su dimensionamiento y sus
controles, el presupuesto restante podrá alojar todos los paquetes INT4
(`int4-stored`), paquetes por capa preparados al usarlos (`int4-jit`) o un conjunto
parcial residente admitido más staging JIT. También hay que contabilizar fuentes
EXL3 retenidas, escalas/índices, workspace de conversión y picos simultáneos de
carga. Más memoria puede permitir INT4 totalmente residente **después** de reservar
el mismo KV; no aumenta automáticamente un contexto fijado ni cambia la identidad
de decode. Actualmente las adaptaciones GLM/MiniMax rechazan esas selecciones INT4
en vez de presentarlas como ejecución EXL3 equivalente.

| Cambio de hardware | Consecuencia para el plan |
|---|---|
| Más memoria utilizable/libre | Puede admitir más slots no fijados o un formato residente disponible; cada rango debe caber. |
| Más ancho de banda de memoria | Cambia el coste de decode y el umbral INT4; exige medidas locales, no escalar rendimiento publicado. El número de CUs no basta. |
| GPU discreta | Presupuesto sobre VRAM libre, contando transferencias host/dispositivo y staging. Más RAM no convierte los pesos en residentes en VRAM. Siguen vigentes soporte de kernels y admisión nativa. |
| Enlace más rápido | Puede reducir el coste de colectivas o transferencias y desplazar el umbral útil de pipeline/desagregación. No habilita topologías ausentes ni suma presupuestos entre nodos. |

Un nodo evita comunicación. Las réplicas conservan un modelo completo por réplica
y atienden peticiones independientes. TP2 reparte los recursos compatibles entre
dos rangos, pero conserva otros replicados. El pipeline Qwen retiene el modelo
completo en cada rango y cuenta el staging de transferencia. El prefill desagregado
separaría procesamiento del prompt y decode y necesitaría transferencia de estado
compatible; sigue pendiente aquí. La selección automática solo considera las
opciones implementadas del proveedor, no todos estos diseños para cada motor.

## Contrato HTTP para orquestadores

Use la autenticación configurada en el servidor para estas rutas.

| Interfaz | Contrato |
|---|---|
| `GET /v1/deployment` | JSON inmutable del plan activo, con `deployment_id`; 404 si no hay callback de plan activo. No se inventa un plan predeterminado. |
| `GET /health` | Incluye `deployment_id` cuando está conectado. Durante drenaje responde HTTP 503. |
| `GET /api/status` | `capacity` incluye números `slots_total`, `slots_busy`, `queue_depth` y `kv_bytes_free` anulable. Los adaptadores actuales publican `null` para KV libre; no significa cero ni memoria libre del dispositivo. |
| `POST /admin/drain` | Cierra admisión de forma idempotente y devuelve `{"status":"draining","drained":false}` hasta completar el trabajo contabilizado, después `drained:true`. Status también expone `drained` tras comenzar el drenaje. |

La admisión de nuevas generaciones tras el drenaje devuelve 503; una cola llena
normal sigue devolviendo 429. Las peticiones en ejecución terminan. La cola común
conserva el trabajo aceptado; DeepSeek mantiene su cancelación previa de trabajo
encolado/diferido. Drenar no termina el proceso. Deje de enviar tráfico, solicite
drenaje, espere a que termine y permita que el gestor de procesos detenga o
sustituya el servidor.

`deployment_id` es el hash canónico del plan (16 dígitos hexadecimales en minúscula),
no un identificador único de proceso/reinicio ni un certificado de fidelidad.
Planes idénticos pueden mantener el ID tras reiniciar; volver a medir puede
cambiarlo. Úselo junto con salud y ciclo de vida del proceso. `/v1/capabilities`
sigue describiendo las políticas API ejecutadas; deployment describe colocación
y presupuestos.

Los controles individuales están en [toggles](toggles.md). Al validar un cambio,
mantenga modelo, perfil, contexto, slots e identidad numérica; compare todos los
IDs de token y logits/estado antes de medir la carga completa.
