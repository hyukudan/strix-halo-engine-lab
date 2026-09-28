# Laboratorio de compilación offline gfx1151

Objetivo: demostrar que podemos generar objetos AMDGPU para Strix Halo sin
tener la GPU presente, inspeccionar su metadata/ISA y fijar el toolchain.

`probe.hip` no es un benchmark. Solo comprueba el pipeline:

1. frontend HIP;
2. target `--offload-arch=gfx1151`;
3. emisión de objeto AMDGPU;
4. lectura de target, símbolos y recursos;
5. desensamblado agregado mediante las herramientas de `../../tools`.

La ausencia de hardware impide validar ejecución, corrección del driver,
rendimiento, ocupación dinámica, memoria UMA y estabilidad. Un objeto que
compila no demuestra que sea rápido ni siquiera que ejecute correctamente.

## Toolchain provisional verificado

La imagen runtime de Halogen no contiene los headers C++ necesarios y no sirve
como entorno de desarrollo completo. La prueba correcta usa la imagen pública
`rocm/dev-ubuntu-24.04:7.2.4`, fijada por digest:

`sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054`

Comando ejecutado dentro del contenedor:

```sh
hipcc -std=c++20 -O3 --offload-arch=gfx1151 --cuda-device-only \
  -c probe.hip -o out/probe.gfx1151.o
```

El bundle contiene un target `hipv4-amdgcn-amd-amdhsa--gfx1151`. Tras separar
el device object, el ELF declara AMDGPU-HSA ABI v4, flag `gfx1151`; el kernel
`saxpy_probe` declara wave32, 14 SGPR, 6 VGPR, 0 LDS y 0 spills.

## Salidas verificadas

- `out/probe.gfx1151.o`, SHA-256
  `3CE334B698C6BCA3D824637ECC7E712350308B47C3AC672CF71824A6072FC562`;
- `out/probe.gfx1151.hsaco`, 4.976 bytes, SHA-256
  `70C2BCDBAD899D22786ED065F8F4F546F676D7924F1D9E5F166426AA3CC19E2F`;
- cabecera ELF con target AMDGPU;
- metadata que declare `amdgcn-amd-amdhsa--gfx1151`;
- resumen de recursos e ISA.

Los binarios generados en `out/` son artefactos fijados por hash y no se
consideran resultados de rendimiento. La reproducibilidad bit a bit requiere
todavía una segunda compilación limpia y comparación normalizada.
