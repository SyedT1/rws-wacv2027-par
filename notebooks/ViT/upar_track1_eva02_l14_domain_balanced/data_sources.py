"""Local UPAR image/ZIP reuse helpers; embedded in the standalone notebook."""
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


def portable_path(value):
    return str(value).replace("\\", "/").strip("/")


def index_local_assets(roots):
    """Scan attached inputs once, without following directory symlinks."""
    images = defaultdict(list)
    archives = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for directory, _, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                path = Path(directory) / filename
                key = str(path.absolute())
                if key in seen:
                    continue
                seen.add(key)
                suffix = path.suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
                    images[filename].append(path)
                elif suffix == ".zip":
                    archives.append(path)
    return images, sorted(archives)


def source_variants(relative, peta_mapping):
    relative = portable_path(relative)
    source, remainder = relative.split("/", 1)
    variants = [relative]
    if source == "Market1501":
        variants.extend([f"Market-1501-v15.09.15/{remainder}", remainder])
    elif source == "PA100k":
        variants.extend([f"PA100K/{remainder}", f"PA-100K/{remainder}"])
        if remainder.startswith("release_data/"):
            variants.append(remainder)
        else:
            variants.append(f"release_data/{remainder}")
    elif source == "PETA" and relative in peta_mapping:
        original = portable_path(peta_mapping[relative])
        variants.append(original)
        variants.append(original.removeprefix("PETA/"))
    return list(dict.fromkeys(variants))


def choose_suffix_match(candidates, variants):
    # Prefer the full challenge path, then known original archive layouts.
    # Never guess by basename alone: PETA originals can share basenames.
    for variant in variants:
        matches = [candidate for candidate in candidates
                   if portable_path(candidate) == variant
                   or portable_path(candidate).endswith("/" + variant)]
        if matches:
            return sorted(matches, key=str)[0]
    return None


def reuse_local_source(data_root, relative_paths, image_index, archives, peta_mapping):
    """Copy only missing annotated images; never import external labels."""
    data_root = Path(data_root)
    relative_paths = list(relative_paths)
    for relative in relative_paths:
        value = str(relative).replace("\\", "/")
        if PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or ":" in value:
            raise ValueError(f"Unsafe annotation path: {relative!r}")
    absent = {portable_path(path) for path in relative_paths
              if not (data_root / portable_path(path)).is_file()}
    copied = 0
    for relative in sorted(absent):
        variants = source_variants(relative, peta_mapping)
        candidates = []
        for basename in {PurePosixPath(variant).name for variant in variants}:
            candidates.extend(image_index.get(basename, []))
        original = choose_suffix_match(candidates, variants)
        if original is not None:
            target = data_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            absent.remove(relative)
            copied += 1
    for archive_path in archives:
        if not absent:
            break
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = defaultdict(list)
                for info in archive.infolist():
                    if not info.is_dir():
                        members[PurePosixPath(portable_path(info.filename)).name].append(info.filename)
                for relative in sorted(absent):
                    variants = source_variants(relative, peta_mapping)
                    candidates = []
                    for basename in {PurePosixPath(variant).name for variant in variants}:
                        candidates.extend(members.get(basename, []))
                    member = choose_suffix_match(candidates, variants)
                    if member is None:
                        continue
                    # Destination comes only from validated annotations, never ZIP paths.
                    target = data_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".part")
                    with archive.open(member) as original, temporary.open("wb") as output:
                        shutil.copyfileobj(original, output)
                    temporary.replace(target)
                    absent.remove(relative)
                    copied += 1
        except (zipfile.BadZipFile, EOFError) as error:
            print(f"Skipping invalid local ZIP {archive_path}: {error}")
    return copied


def local_reuse_self_check():
    """Small real filesystem checks, before any network access."""
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs, output = root / "inputs", root / "output"
        image = inputs / "old-output/upar_2027/data/Market1501/bounding_box_train/a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"market-fixture")
        with zipfile.ZipFile(inputs / "originals.zip", "w") as archive:
            archive.writestr("release_data/b.jpg", b"pa-fixture")
            archive.writestr("PETA/group/0001.png", b"peta-fixture")
            archive.writestr("PETA/other/0001.png", b"wrong-peta")
            archive.writestr("../../outside.txt", b"unsafe-unused-member")
        images, archives = index_local_assets([inputs])
        paths = ["Market1501/bounding_box_train/a.jpg", "PA100k/release_data/b.jpg", "PETA/mapped.png"]
        mapping = {"PETA/mapped.png": "PETA/group/0001.png"}
        assert reuse_local_source(output, paths, images, archives, mapping) == 3
        assert [(output / path).read_bytes() for path in paths] == [b"market-fixture", b"pa-fixture", b"peta-fixture"]
        assert reuse_local_source(output, paths, images, archives, mapping) == 0
        assert not (root / "outside.txt").exists()
        assert choose_suffix_match(["unrelated/a.jpg"], ["bounding_box_train/a.jpg"]) is None
        with zipfile.ZipFile(inputs / "market.zip", "w") as archive:
            archive.writestr("Market-1501-v15.09.15/query/c.jpg", b"original-market")
        images, archives = index_local_assets([inputs])
        assert reuse_local_source(output, ["Market1501/query/c.jpg"], images, archives, {}) == 1
        assert (output / "Market1501/query/c.jpg").read_bytes() == b"original-market"
    print("Local image/ZIP reuse, PETA mapping, rerun and path checks passed")
