#!/bin/bash
# Prepares the native FP8 dense weights for a GPU validation on the nodes. Run on the workstation that holds the
# original checkpoint (bash, e.g. Git Bash) once its download is complete; it is never run automatically.
#   NODES="node1 node2" [NODE_USER=user] [DEST=models/ds41-native-fp8] [PY=python3]
#     tools/ds41_native_fp8/copy-to-nodes.sh SRC_DIR EXTRACT_DIR
# 1. extract.py copies only the dense FP8 sites (attention wq_a/wq_b/wkv/wo_a/wo_b + shared expert w1/w3/w2, codes and
#    32 x 32 exponents) of the 40 layers: 40 files of 162,163,200 bytes + index, 6.49 GB (not the 280+ GB checkpoint);
# 2. verifies the extract byte for byte against the source (--check);
# 3. on each node: requires free space >= 2 x the extract, then copies it to ~/$DEST
#    (scp to the first node, rsync from there at low priority; every file is checked by sha256 after the copy).
# Nothing is executed from the model directories; Python runs isolated (-I) from this tools directory.
set -euo pipefail
SRC=${1:?usage: copy-to-nodes.sh SRC_DIR EXTRACT_DIR}
OUT=${2:?usage: copy-to-nodes.sh SRC_DIR EXTRACT_DIR}
PY=${PY:-python3}
NODES=${NODES:?set NODES to the node host names, e.g. NODES="node1 node2"}
DEST=${DEST:-models/ds41-native-fp8}
at=${NODE_USER:+$NODE_USER@}
here=$(cd "$(dirname "$0")" && pwd)

"$PY" -I "$here/extract.py" "$SRC" "$OUT" --layers 0-39
"$PY" -I "$here/extract.py" "$SRC" "$OUT" --layers 0-39 --check
bytes=$(du -sb "$OUT" | cut -f1)
need=$((2 * bytes / 1024))
( cd "$OUT" && sha256sum dense-fp8-*.safetensors model.safetensors.index.json > SHA256SUMS )
first=${NODES%% *}
for h in $NODES; do
  free=$(ssh -o BatchMode=yes "$at$h" "df -k --output=avail ~ | tail -1")
  echo "$h: free $((free / 1024 / 1024)) GiB, extract $((bytes / 1024 / 1024)) MiB"
  if [ "$free" -lt "$need" ]; then echo "ABORT: not enough free disk on $h"; exit 3; fi
done
# workstation -> first node (scp), then first node -> the others over the node link (rsync at low priority).
ssh -o BatchMode=yes "$at$first" "mkdir -p ~/$DEST"
scp -q "$OUT"/* "$at$first:$DEST/"
for h in $NODES; do
  [ "$h" = "$first" ] || ssh -o BatchMode=yes "$at$first" "nice -n 19 ionice -c3 rsync -a --partial ~/$DEST/ $h:$DEST/"
  ssh -o BatchMode=yes "$at$h" "cd ~/$DEST && sha256sum -c --quiet SHA256SUMS && echo $h: extract verified"
done
