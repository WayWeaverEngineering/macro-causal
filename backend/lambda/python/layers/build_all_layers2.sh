#!/usr/bin/env bash
set -euo pipefail

# Minimal Lambda layer builder for CodeBuild ShellStep
# - Builds all immediate subdirectories as layers
# - Installs deps into ./python
# - Zips ONLY the top-level python/ dir
# - Exits on error

# Config
LAYERS_ROOT="${LAYERS_ROOT:-.}"              # set to "layers" if your layers live under ./layers
PIP_CMD="${PIP_CMD:-python3 -m pip}"         # override if needed, e.g. PIP_CMD="pip3"

echo "=== Building Lambda layers ==="
echo "Root: $LAYERS_ROOT"
echo "Pip : $PIP_CMD"
echo

# Optional: verify tools (zip is required; unzip is optional)
command -v zip >/dev/null 2>&1 || { echo "ERROR: zip not found"; exit 1; }
if command -v unzip >/dev/null 2>&1; then HAS_UNZIP=true; else HAS_UNZIP=false; fi

# Iterate immediate subdirectories (each is a layer)
shopt -s nullglob
LAYER_DIRS=("$LAYERS_ROOT"/*/)
if [ ${#LAYER_DIRS[@]} -eq 0 ]; then
  echo "No layer directories found under: $LAYERS_ROOT"
  exit 1
fi

for layer_dir in "${LAYER_DIRS[@]}"; do
  layer_dir="${layer_dir%/}"                  # strip trailing slash
  layer_name="$(basename "$layer_dir")"
  echo "=== Layer: $layer_name ==="

  # Clean previous artifacts
  rm -rf "$layer_dir/python" "$layer_dir/layer.zip" "$layer_dir/python.zip" || true

  # Create python/ and install deps
  mkdir -p "$layer_dir/python"
  if [ -f "$layer_dir/requirements.txt" ]; then
    echo "Installing requirements.txt..."
    $PIP_CMD install --upgrade pip --quiet
    $PIP_CMD install -r "$layer_dir/requirements.txt" --target "$layer_dir/python" --no-cache-dir --quiet
  else
    echo "No requirements.txt found (ok if you only ship src/)."
  fi

  # Copy your own modules if present
  if [ -d "$layer_dir/src" ]; then
    cp -R "$layer_dir/src/." "$layer_dir/python/"
  fi

  # Zip ONLY the python/ directory (avoid zipping ".")
  (
    cd "$layer_dir"
    zip -r9 layer.zip python \
      -x "python/**/__pycache__/*" \
         "python/**/*.pyc" \
         "python/**/.pytest_cache/*" \
         "python/**/tests/*" \
         "python/**/test/*" \
         "python/**/.DS_Store" \
         "python/**/*.dist-info/RECORD" \
      >/dev/null
  )

  # Optional verification (if unzip is available)
  if $HAS_UNZIP; then
    if ! unzip -l "$layer_dir/layer.zip" | awk '{print $4}' | grep -q '^python/'; then
      echo "ERROR: $layer_name layer.zip does not contain top-level 'python/'"
      exit 1
    fi
  fi

  # Size info
  ZIP_SIZE=$(du -h "$layer_dir/layer.zip" | cut -f1)
  echo "✓ Built $layer_name -> $layer_dir/layer.zip ($ZIP_SIZE)"
  echo
done

echo "All layers built successfully."
