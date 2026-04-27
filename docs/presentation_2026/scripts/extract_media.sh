#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMG="$ROOT/img"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PPTX="$ROOT/../presentation/ConcreteNet talk.pptx"
TAHERI="$HOME/Documents/Anish Goyal's Vault/03. References/Research/Hossein Taheri"

echo "[1/2] Extracting embedded media from PPTX..."
mkdir -p "$TMP/pptx"
unzip -o -j "$PPTX" 'ppt/media/*' -d "$TMP/pptx" >/dev/null
# Inspect then copy into sensible subfolders
ls -la "$TMP/pptx" | head -20
mkdir -p "$IMG/machines" "$IMG/gpr" "$IMG/defects" "$IMG/misc"
cp -n "$TMP/pptx"/* "$IMG/misc/" 2>/dev/null || true
echo "  -> staged PPTX media into img/misc/ (manually re-sort interesting ones into gpr/ machines/ defects/)"

echo "[2/2] Extracting images from Hossein Taheri reference PDFs..."
mkdir -p "$IMG/gpr" "$IMG/defects"
cd "$TAHERI"
for pdf in \
  "Snells_Law_GPR.pdf" \
  "Nondestructive Evaluation of Structural Defects in Concrete Slabs.pdf" \
  "Deep-Learning-Based Method for Estimating Permittivity of Ground-Penetrating Radar Targets.pdf" \
  "Deep learning-based pavement subsurface distress detection via ground penetrating radar data.pdf" \
  "1D-CNNs for autonomous defect detection in bridge decks using ground penetrating radar.pdf" \
  ; do
  if [[ -f "$pdf" ]]; then
    base="$(basename "$pdf" .pdf | tr ' ' '_' | cut -c1-40)"
    echo "  - $pdf -> $base"
    pdfimages -all -p "$pdf" "$TMP/${base}" || true
  fi
done

# Collect all extracted images
mkdir -p "$IMG/gpr/from_pdfs"
find "$TMP" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.jp2' -o -iname '*.ppm' -o -iname '*.pbm' \) -exec cp -n {} "$IMG/gpr/from_pdfs/" \;

echo
echo "Done. Review the staged images:"
echo "  $IMG/misc/     (from PPTX — re-sort as needed)"
echo "  $IMG/gpr/from_pdfs/  (from Taheri PDFs — pick the best ones)"
