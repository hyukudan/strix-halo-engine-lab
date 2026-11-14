#!/bin/bash
R=$HOME/ai-cluster-lab/dspf; case $1 in
6) bash $R/pr.sh 8192 app-ring r1-ring; bash $R/pr.sh 8192 app-base r1-base; bash $R/pr.sh 8192 app-ring r2-ring;;
7) bash $R/pr.sh 8192 app-new r2-new; bash $R/pr.sh 8192 app-ring r3-ring; bash $R/pr.sh 8192 app-base r3-base;;
8) bash $R/pr.sh 11594 app-ring m1-ring; bash $R/pr.sh 11594 app-base m1-base;;
9) bash $R/pr.sh 11594 app-ring m2-ring; bash $R/pr.sh 11594 app-new m2-new;;
esac; echo "T$1 DONE"
