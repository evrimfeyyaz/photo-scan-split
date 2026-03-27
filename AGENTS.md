## Learned User Preferences

- Prefers auto-incrementing file counters over overwriting existing files
- Wants verbose/debug flags on CLI commands for troubleshooting
- Prefers pure-Python solutions with minimal system-level dependencies

## Learned Workspace Facts

- Project: Python CLI tool (`photo-scan-split`) that scans via WiFi and splits photos automatically
- Scanner: Canon G3020 series, connected over WiFi, has a dark scanner lid
- Scanner protocol: eSCL (AirScan) over HTTP, discovered via mDNS (`_uscan._tcp.local.`)
- Tech stack: Python, Click, zeroconf, requests, lxml, opencv-python-headless, Pillow, numpy
- Dark scanner lid requires auto-detection of background brightness (Otsu's method) instead of a hardcoded threshold
- Photo splitting uses content discontinuity detection (histogram comparison) + Hough line detection for touching photos
- Three CLI commands: `scan` (full pipeline), `discover` (list scanners), `split` (split existing image)
- Entry points: `photo-scan-split` console script and `python -m photo_scan_split`
