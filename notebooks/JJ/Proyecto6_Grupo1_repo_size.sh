#!/bin/bash

OUTPUT=~/Proyectos/Proyecto6_Grupo1.20260306100447V/Proyecto6_Grupo1_repo_sizes_$(date +%Y%m%d%H%M%S).txt

for branch in $(git branch -r | grep -v '\->' | sed 's/origin\///'); do
  echo ""
  echo "===== RAMA: $branch ====="
  git branch -r --sort=-committerdate \
    --format="%(refname:short) — %(committerdate:format:%Y%m%d%H%M%S) — %(subject)" \
    | grep "origin/$branch"

  echo ""
  # git ls-tree -r --long origin/$branch | awk '{printf "%8s KB  %s\n", int($4/1024)+1, $5}'
  git ls-tree -r --long origin/$branch | awk '{printf "%08d bytes\t%s\n", $4, $5}'
done > "$OUTPUT"

echo "✅ Guardado en: $OUTPUT"