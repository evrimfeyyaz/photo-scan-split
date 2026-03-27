# photo-scan-split

A CLI tool that scans photos from a WiFi scanner (via the eSCL/AirScan protocol) and automatically splits them into individual images.

Place multiple photos on your flatbed scanner, run one command, and get separate image files — even if the photos are touching edge-to-edge.

## How It Works

1. **Discovers** your scanner on the local network via mDNS/Bonjour
2. **Scans** the full flatbed using the eSCL (AirScan) HTTP protocol — no drivers needed
3. **Detects** individual photos using watershed segmentation with distance transform (OpenCV)
4. **Saves** each photo as a separate file in your chosen format

## Requirements

- Python 3.10+
- A WiFi scanner that supports eSCL/AirScan (most modern Canon, HP, Brother, Epson scanners do)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/photo-scan-split.git
cd photo-scan-split

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Full Pipeline: Scan and Split

```bash
# Auto-discover scanner, scan at 300 DPI, split and save as JPEG
photo-scan-split scan

# Scan at 600 DPI and save as PNG
photo-scan-split scan --dpi 600 --format png

# Save to a specific directory with a custom filename prefix
photo-scan-split scan -o ~/Photos/scanned -p vacation

# Skip auto-discovery by providing the scanner URL directly
photo-scan-split scan --scanner http://192.168.1.42:80/eSCL/

# Save the raw scan without splitting
photo-scan-split scan --no-split
```

### Split an Existing Image

If you already have a scanned image, you can split it without scanning again:

```bash
photo-scan-split split scan.jpg -o ./output
photo-scan-split split scan.png --format tiff --prefix family
```

### Discover Scanners

List available eSCL scanners on your network:

```bash
photo-scan-split discover
```

## Options Reference

### `photo-scan-split scan`

| Option | Default | Description |
|---|---|---|
| `-o, --output-dir` | `.` | Directory to save split photos |
| `-f, --format` | `jpeg` | Output format: `jpeg`, `png`, or `tiff` |
| `-d, --dpi` | `300` | Scan resolution |
| `-s, --scanner` | (auto) | Scanner URL to skip auto-discovery |
| `-p, --prefix` | `photo` | Filename prefix |
| `--no-split` | off | Save the raw scan without splitting |
| `--min-area` | `3.0` | Minimum photo area as % of total scan |
| `--threshold` | `230` | Background brightness threshold (0-255) |
| `-t, --timeout` | `5.0` | Scanner discovery timeout in seconds |

### `photo-scan-split split`

| Option | Default | Description |
|---|---|---|
| `-o, --output-dir` | `.` | Directory to save split photos |
| `-f, --format` | `jpeg` | Output format: `jpeg`, `png`, or `tiff` |
| `-p, --prefix` | `photo` | Filename prefix |
| `--min-area` | `3.0` | Minimum photo area as % of total scan |
| `--threshold` | `230` | Background brightness threshold (0-255) |

## Troubleshooting

### Scanner not found

- Make sure your scanner is on and connected to the same WiFi network as your computer.
- Try increasing the discovery timeout: `photo-scan-split discover --timeout 15`
- Check if your scanner is reachable by visiting `http://<scanner-ip>/eSCL/ScannerCapabilities` in a browser.
- If auto-discovery doesn't work, provide the URL directly with `--scanner`.

### Photos not detected correctly

- If photos blend into the background, try lowering `--threshold` (e.g., `200`).
- If small regions are being picked up as photos, increase `--min-area`.
- If a dark scanner lid is causing issues, try raising `--threshold` (e.g., `240`).

## License

MIT
