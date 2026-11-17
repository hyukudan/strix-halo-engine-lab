# Declaración de la topología de nodos

*[English](topology.md)*

Una ejecución en varios nodos se describe igual que un dispositivo: declarada en un fichero o descubierta, resuelta
por una única función compartida, impresa en la configuración efectiva y nunca supuesta por un modelo. La
declaración de dispositivos (`strix-devices-v1`, [despliegue](../deployment.es.md)) dice qué GPU puede usar una
máquina; la topología (`strix-topology-v1`, `include/strix/device/topology.hpp`) dice qué nodos participan en una
ejecución, cuánto cabe en cada uno, qué velocidad tiene y qué los une. Un solo nodo y el par de dos nodos de hoy son
solo dos casos.

## El fichero

Un fichero describe todo el clúster y es idéntico en cada nodo. Una entrada de nodo es un rango, es decir, un
proceso de GPU; una máquina con dos GPU son dos entradas con el mismo `host`, unidas por un enlace PCIe.

```toml
schema = "strix-topology-v1"
name = "two-node-usb4"                 # opcional, solo para mostrar

[nodes.a]                              # la clave de la tabla es el id del nodo
rank = 0                               # rangos 0..N-1, únicos y contiguos
host = "node-a"                        # opcional: se compara con el nombre de esta máquina
profile = "strix-halo-128"             # un modelo del catálogo (strix-device-plan --list), o "custom"
memory_gib = 124                       # opcional: memoria de dispositivo usable para la colocación
bandwidth_gbytes_ps = 243              # opcional: velocidad de memoria medida
fp16_tflops = 59.4                     # opcional: rendimiento FP16 denso
units = 40                             # opcional: número de CU / SM
roles = ["prefill", "decode"]          # opcional; ausente = todos los papeles multinodo

[nodes.b]
rank = 1
profile = "strix-halo-128"

[links.ab]
between = ["a", "b"]
class = "usb4"                         # las mismas clases que la declaración de dispositivos
gbps_measured = 36.6                   # opcional: Gbit/s útiles medidos; sustituye a la clase nominal
rtt_us_measured = 30                   # opcional
port = "usb4-0"                        # opcional: recurso compartido; dos enlaces entre los mismos nodos necesitan dos puertos
```

Un nodo puede diferir de los demás en cualquier dato: memoria, velocidad de memoria, cómputo y enlaces. Cada valor se
imprime con su origen: `declared`, `profile`, `observed` (descubrimiento), `legacy` o `unknown`. Desconocido no es una
suposición: un planificador que lo necesite se niega y da el motivo.

Se rechaza nombrando la clave: una clave, perfil, clase o papel desconocidos; un rango que falta, repetido o fuera de
intervalo; un perfil `custom` sin `memory_gib`; un enlace con un solo extremo, hacia un nodo desconocido o hacia sí
mismo; dos enlaces entre los mismos nodos en el mismo puerto; un nodo al que no llega ningún enlace.

## Qué nodo es este proceso

`--node ID` > `STRIX_NODE` > el antiguo `--rank R` (debe nombrar un rango declarado) > el nodo cuyo `host` es el
nombre de esta máquina > el único nodo. En otro caso el arranque se rechaza: "cannot tell which node this process
is". `--topology FICHERO` > `STRIX_TOPOLOGY` elige el fichero; `take_topology_command_line` retira las dos opciones
nuevas de los argumentos para que cualquier binario las acepte, como `--devices`.

## Sin fichero (compatibilidad)

- `--rank R` con un transporte de par (`usb4auto`, `usb4-auto`, `usb4gpu`, `usb4-gpu`, `usb4:...`, `tcp-*`) se
  resuelve a la forma que hoy suponen todas las rutas de par: dos nodos, el rango R local, un enlace de clase `usb4`
  (o velocidad desconocida con tcp), `source=legacy`. La colocación no cambia: las rutas conservan su reparto propio
  hasta que lean la topología.
- Sin rango y sin fichero es un solo nodo (`source=single`).

## Descubrimiento

`observe(topology, rank, ObservedNode)` recibe lo que midió un nodo (arquitectura, memoria usable, unidades), de la
sonda local o del saludo de un par. Rellena lo que el fichero dejó desconocido y rechaza una arquitectura declarada
distinta o una memoria declarada mayor que la que informa el nodo.

## Impresión y consumidores

`topology_line()` imprime una línea junto al recibo `device:` (formato en la [versión inglesa](topology.md)).
`topology_json()` es el objeto `strix-topology-v1` para la configuración efectiva en JSON; `topology_id` resume el
clúster sin el rango local, así que todos los nodos de una ejecución imprimen el mismo id. `to_declaration()` da los
mismos nodos y enlaces como una declaración de dispositivos, de modo que el grafo de enlaces y el planificador de
rutas compartidos (`include/strix/deploy/link.hpp`, `route.hpp`) leen el clúster con el único modelo de enlaces de
`include/strix/device/link_class.hpp`.

La topología no contiene conocimiento de modelos ni decide la colocación. El planificador compartido convierte
(topología, tabla de costes del modelo, nivel) en un plan.

Pruebas: `tests/device_topology_tests.cpp` (1, 2 iguales, 2 distintos, 3 y 4 nodos, ficheros inválidos, forma
antigua, identidad, descubrimiento, puente al grafo de enlaces), objetivo `strix-device-topology-tests`.

## Enlazado hoy

- Puerta de dispositivo: `bind_hip_main` de cada binario HIP toma `--topology FICHERO` / `--node ID` e imprime
  `topology:` junto a `device:` cuando hay fichero (o un `--rank R` antiguo en la línea de órdenes). Un fichero que no
  puede situar este proceso detiene el arranque con el motivo.
- Rutas de pareja (Qwen, Gemma 4; `include/strix/pair/link_flags.hpp`): con fichero, el rango, el host del par y el
  transporte que las opciones dejan abiertos salen de él (clases Ethernet -> `tcp`, clases `usb4` -> el valor de la
  familia, PCIe/OCuLink rechazadas); `--rank/--peer/--transport` explícitos siguen mandando. La ruta de pareja tiene
  dos rangos: otro número de nodos se rechaza con el motivo. Sin fichero las opciones antiguas se comportan igual.
- `--print-effective-config` de Qwen lleva el miembro `"topology"` cuando el proceso resolvió una.
- Una sola fuente de velocidad de enlace: `link_class.hpp` es la única tabla de clases (`interconnects()` es una vista).
  Las decisiones calibradas con una velocidad de protocolo la conservan como fila de calibración con nombre (relevo
  por el host 4.400 MB/s; traspaso por corte de capas 118 / 290 / 1.180 / 4.600 MB/s; salida del host 125 MB/s).
  Regla: medida declarada > calibración de la clase > nominal de la clase. Sin fichero, o con el fichero de dos nodos
  `usb4` del laboratorio sin medida, salen exactamente los números anteriores.
- Un solo planificador de colocación para cadenas de capas (`include/strix/pair/chain.hpp`), enlazado a la
  declaración por `include/strix/pair/chain_topology.hpp`: rangos en orden en una línea, memoria por nodo, ritmo de
  prefill escalado por el `fp16_tflops` de cada nodo frente al más rápido, enlaces de la línea al ritmo declarado. Con
  la memoria de todos los nodos declarada, una tubería con estado por capa solo acepta planes cuyas etapas caben: un
  nodo más pequeño recibe menos capas en vez de un arranque fallido. Lo usa la ruta de pareja de Qwen denso (N rangos).
- Equilibrio de etapas sobre N rangos (`include/strix/pair/balance.hpp` `balance_chain`, un vector de costes por
  rango); `balance_split` de dos rangos es su forma N = 2, con el mismo corte bit a bit. El corte de pareja de Qwen MoE
  (fichero de costes medido) escala los costes de cada rango por su velocidad declarada (`stage_cost_scale`); nodos
  iguales o sin declarar escalan por 1,0.
- Colectivas de host sobre N rangos (`include/strix/dist/mesh.hpp`: all-gather / all-reduce ordenados con
  intercambios por parejas) e intercambio de hechos de despliegue sobre N (`exchange_facts`; `exchange_pair_facts` es
  su forma N = 2, mismo registro en el cable).

Pruebas: `tests/device_topology_bindings_tests.cpp`, objetivo `strix-device-topology-bindings-tests`;
`tests/pair_topology_plan_tests.cpp`, objetivo `strix-pair-topology-plan-tests` (1-4 rangos, memoria desigual, un rango
lento, un enlace lento, el plan del laboratorio sin cambios).
