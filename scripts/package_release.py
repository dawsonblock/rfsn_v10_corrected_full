#!/usr/bin/env python3
"""Package Main 27 release into a clean zip file."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main():
    """Package the Main 27 release."""
    root = Path(__file__).parent.parent
    proof_dir = root / "artifacts" / "proof" / "main27"
    manifest_path = proof_dir / "main27_release_manifest.json"

    # Read manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    release_name = manifest["release"]
    title = manifest["title"].replace(" ", "_").lower()

    # Create temp directory for packaging
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pkg_dir = tmp_path / f"rfsn_v10_{release_name}_{title}"
        pkg_dir.mkdir()

        # Copy artifacts
        artifact_dest = pkg_dir / "artifacts" / "proof" / release_name
        artifact_dest.mkdir(parents=True)

        for item in proof_dir.iterdir():
            if item.is_file() and not item.name.endswith(".tmp"):
                shutil.copy2(item, artifact_dest / item.name)

        # Copy core source files (rfsn_v10 package)
        src_dest = pkg_dir / "rfsn_v10"
        shutil.copytree(root / "rfsn_v10", src_dest)

        # Copy benchmark scripts
        bench_dest = pkg_dir / "benchmarks"
        shutil.copytree(root / "benchmarks", bench_dest)

        # Copy validation scripts
        scripts_dest = pkg_dir / "scripts"
        scripts_dest.mkdir(parents=True, exist_ok=True)
        for script in [
            "check_release_integrity.py",
            "generate_summary.py",
            "check_proof_regression.py",
            "compare_proof_runs.py",
        ]:
            src = root / "scripts" / script
            if src.exists():
                shutil.copy2(src, scripts_dest / script)

        # Copy README
        shutil.copy2(root / "README.md", pkg_dir / "README.md")

        # Copy pyproject.toml
        if (root / "pyproject.toml").exists():
            shutil.copy2(root / "pyproject.toml", pkg_dir / "pyproject.toml")

        # Create zip
        zip_path = root / f"rfsn_v10_{release_name}_{title}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in pkg_dir.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(pkg_dir)
                    zf.write(item, arcname)

        print(f"Packaged release to: {zip_path}")
        print(f"Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
