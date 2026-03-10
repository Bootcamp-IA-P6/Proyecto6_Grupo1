#!/bin/bash

WORKDIR=~/Proyectos/Proyecto6_Grupo1.20260306100447V/branches
mkdir -p "$WORKDIR"

for branch in $(git branch -r | grep -v '\->' | sed 's/origin\///'); do
  echo ""
  echo "===== Descargando RAMA: $branch ====="
  
  # Nombre seguro para el archivo (reemplaza / por _)
  SAFE_NAME=$(echo "$branch" | sed 's/\//_/g')
  ZIPFILE="$WORKDIR/Proyecto6_Grupo1_${SAFE_NAME}_$(date +%Y%m%d%H%M%S).zip"
  
  git archive --format=zip origin/$branch -o "$ZIPFILE"
  
  echo "✅ Guardado en: $ZIPFILE"
done

echo ""
echo "===== RESUMEN ====="
ls -lh "$WORKDIR"
