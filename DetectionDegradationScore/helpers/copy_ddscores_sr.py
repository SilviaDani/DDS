import os
import shutil
from pathlib import Path

SOURCE_ROOT = Path("ddscores_analysis/mapping/01_flir_sr")
DEST_ROOT = Path("ddscores_analysis/mapping/01_flir_sr")

def get_latest_timestamp_folder(root: Path):
    """Trova l’ultima cartella generata dallo script multi_dds_calculator_sr.py"""
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError("Nessuna cartella timestamp trovata in:", root)
    return max(subdirs, key=os.path.getmtime)

def copy_ddscores(latest_folder: Path, dest_root: Path):
    for split in ["train", "val", "test"]:
        src_file = latest_folder / split / "ddscores.json"
        dest_dir = dest_root / split
        dest_file = dest_dir / "ddscores.json"

        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Copio: {src_file} → {dest_file}")
        shutil.copy2(src_file, dest_file)

    print("\n✔ Copia completata!")

if __name__ == "__main__":
    print("Cerco l'ultima cartella timestamp...")
    latest = get_latest_timestamp_folder(SOURCE_ROOT)
    print("Ultima trovata:", latest)

    copy_ddscores(latest, DEST_ROOT)
