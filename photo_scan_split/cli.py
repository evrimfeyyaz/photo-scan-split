"""CLI for photo-scan-split."""

from __future__ import annotations

import io
import re
from pathlib import Path

import click
from PIL import Image

from photo_scan_split.scanner import (
    ScannerError,
    discover_scanners,
    get_capabilities,
    scan,
)
from photo_scan_split.splitter import split_photos

FORMAT_EXTENSIONS = {
    "jpeg": ".jpg",
    "png": ".png",
    "tiff": ".tiff",
}

PILLOW_FORMAT_MAP = {
    "jpeg": "JPEG",
    "png": "PNG",
    "tiff": "TIFF",
}


def _status(msg: str) -> None:
    click.echo(msg, err=True)


def _find_next_counter(output_dir: Path, prefix: str, ext: str) -> int:
    """Find the highest existing counter for the given prefix/ext and return the next one."""
    pattern = re.compile(re.escape(prefix) + r"_(\d+)" + re.escape(ext) + "$")
    max_counter = 0
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            m = pattern.match(path.name)
            if m:
                max_counter = max(max_counter, int(m.group(1)))
    return max_counter + 1


def _save_photos(
    photos: list,
    output_dir: Path,
    prefix: str,
    fmt: str,
) -> list[Path]:
    """Save detected photos and return the list of saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = FORMAT_EXTENSIONS[fmt]
    pil_fmt = PILLOW_FORMAT_MAP[fmt]
    saved: list[Path] = []

    start = _find_next_counter(output_dir, prefix, ext)

    for i, photo in enumerate(photos):
        counter = start + i
        filename = f"{prefix}_{counter:03d}{ext}"
        path = output_dir / filename
        save_kwargs = {}
        if pil_fmt == "JPEG":
            save_kwargs["quality"] = 95
        photo.image.save(path, format=pil_fmt, **save_kwargs)
        saved.append(path)
        _status(f"  Saved {path}")

    return saved


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="photo-scan-split")
def cli() -> None:
    """Scan photos from a WiFi scanner and split them into individual images."""


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--timeout", "-t", default=10.0, show_default=True,
    help="Seconds to wait for scanner announcements.",
)
def discover(timeout: float) -> None:
    """Discover eSCL scanners on the local network."""
    _status(f"Searching for scanners ({timeout}s timeout)...")
    scanners = discover_scanners(timeout=timeout)

    if not scanners:
        _status("No scanners found.")
        raise SystemExit(1)

    for s in scanners:
        try:
            caps = get_capabilities(s.base_url)
            model = caps.make_and_model or "Unknown model"
            resolutions = sorted(set(caps.x_resolutions) & set(caps.y_resolutions))
        except Exception:
            model = "Could not query capabilities"
            resolutions = []

        click.echo(f"\n  Name:        {s.name}")
        click.echo(f"  URL:         {s.base_url}")
        click.echo(f"  Model:       {model}")
        if resolutions:
            click.echo(f"  Resolutions: {', '.join(str(r) for r in resolutions)} DPI")


# ---------------------------------------------------------------------------
# split (offline, from an existing image file)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output-dir", "-o", type=click.Path(), default=".",
    show_default=True, help="Directory to save split photos.",
)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["jpeg", "png", "tiff"]),
    default="jpeg", show_default=True, help="Output image format.",
)
@click.option(
    "--prefix", "-p", default="photo", show_default=True,
    help="Filename prefix for saved photos.",
)
@click.option(
    "--min-area", default=3.0, show_default=True,
    help="Minimum photo area as %% of total scan area.",
)
@click.option(
    "--threshold", default=230, show_default=True,
    help="Background brightness threshold (0-255).",
)
def split(
    image_path: str,
    output_dir: str,
    fmt: str,
    prefix: str,
    min_area: float,
    threshold: int,
) -> None:
    """Split an existing scanned image into individual photos."""
    _status(f"Loading {image_path}...")
    image = Image.open(image_path)
    image.load()

    _status("Detecting photos...")
    photos = split_photos(image, min_area_pct=min_area, bg_threshold=threshold)

    if not photos:
        _status("No photos detected in the image.")
        raise SystemExit(1)

    _status(f"Found {len(photos)} photo(s). Saving...")
    saved = _save_photos(photos, Path(output_dir), prefix, fmt)
    _status(f"\nDone! {len(saved)} photo(s) saved to {output_dir}/")


# ---------------------------------------------------------------------------
# scan (full pipeline: discover -> scan -> split -> save)
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--output-dir", "-o", type=click.Path(), default=".",
    show_default=True, help="Directory to save split photos.",
)
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["jpeg", "png", "tiff"]),
    default="jpeg", show_default=True, help="Output image format.",
)
@click.option(
    "--dpi", "-d", default=300, show_default=True,
    help="Scan resolution in DPI.",
)
@click.option(
    "--scanner", "-s", default=None,
    help="Scanner URL (skip auto-discovery).",
)
@click.option(
    "--prefix", "-p", default="photo", show_default=True,
    help="Filename prefix for saved photos.",
)
@click.option(
    "--no-split", is_flag=True, default=False,
    help="Save the raw scan without splitting.",
)
@click.option(
    "--min-area", default=3.0, show_default=True,
    help="Minimum photo area as %% of total scan area.",
)
@click.option(
    "--threshold", default=230, show_default=True,
    help="Background brightness threshold (0-255).",
)
@click.option(
    "--timeout", "-t", default=10.0, show_default=True,
    help="Scanner discovery timeout in seconds.",
)
@click.option(
    "--once", is_flag=True, default=False,
    help="Scan once and exit instead of looping.",
)
def scan_cmd(
    output_dir: str,
    fmt: str,
    dpi: int,
    scanner: str | None,
    prefix: str,
    no_split: bool,
    min_area: float,
    threshold: int,
    timeout: float,
    once: bool,
) -> None:
    """Scan photos and automatically split them into individual images.

    By default the command loops: after each scan it waits for you to
    place new photos and press Enter to scan again.  Press Ctrl+C to
    stop.  Use --once to scan a single time and exit.
    """
    # --- Discover or use provided scanner URL ---
    if scanner:
        base_url = scanner.rstrip("/") + "/"
        _status(f"Using scanner at {base_url}")
    else:
        _status(f"Searching for scanners ({timeout}s timeout)...")
        scanners = discover_scanners(timeout=timeout)
        if not scanners:
            _status("No scanners found on the network. Use --scanner to specify a URL manually.")
            raise SystemExit(1)

        if len(scanners) == 1:
            base_url = scanners[0].base_url
            _status(f"Found scanner: {scanners[0].name}")
        else:
            _status(f"Found {len(scanners)} scanners:")
            for i, s in enumerate(scanners, 1):
                _status(f"  {i}. {s.name} ({s.base_url})")
            choice = click.prompt(
                "Select scanner", type=click.IntRange(1, len(scanners)), err=True,
            )
            base_url = scanners[choice - 1].base_url

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    batch = 0

    while True:
        batch += 1
        if batch > 1:
            _status("")

        # --- Scan ---
        try:
            image_bytes = scan(
                base_url=base_url,
                dpi=dpi,
                on_status=_status,
            )
        except ScannerError as exc:
            _status(f"Scan failed: {exc}")
            if once:
                raise SystemExit(1)
            _status("You can try again.")
            try:
                click.prompt(
                    "\nPress Enter to re-scan (Ctrl+C to quit)",
                    default="", show_default=False, prompt_suffix="", err=True,
                )
            except (KeyboardInterrupt, EOFError):
                _status("\nStopped.")
                return
            continue

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        # --- Optionally skip splitting ---
        if no_split:
            ext = FORMAT_EXTENSIONS[fmt]
            raw_path = out_path / f"{prefix}_raw{ext}"
            pil_fmt = PILLOW_FORMAT_MAP[fmt]
            save_kwargs = {"quality": 95} if pil_fmt == "JPEG" else {}
            image.save(raw_path, format=pil_fmt, **save_kwargs)
            _status(f"Raw scan saved to {raw_path}")
        else:
            # --- Split ---
            _status("Detecting photos...")
            photos = split_photos(image, min_area_pct=min_area, bg_threshold=threshold)

            if not photos:
                _status("No individual photos detected. Saving raw scan instead.")
                ext = FORMAT_EXTENSIONS[fmt]
                raw_path = out_path / f"{prefix}_raw{ext}"
                pil_fmt = PILLOW_FORMAT_MAP[fmt]
                save_kwargs = {"quality": 95} if pil_fmt == "JPEG" else {}
                image.save(raw_path, format=pil_fmt, **save_kwargs)
                _status(f"Raw scan saved to {raw_path}")
            else:
                _status(f"Found {len(photos)} photo(s). Saving...")
                saved = _save_photos(photos, out_path, prefix, fmt)
                _status(f"\nDone! {len(saved)} photo(s) saved to {output_dir}/")

        if once:
            return

        try:
            click.prompt(
                "\nPlace next photos and press Enter to scan (Ctrl+C to quit)",
                default="", show_default=False, prompt_suffix="", err=True,
            )
        except (KeyboardInterrupt, EOFError):
            _status("\nStopped.")
            return


# Register scan_cmd under the name "scan"
scan_cmd.name = "scan"
