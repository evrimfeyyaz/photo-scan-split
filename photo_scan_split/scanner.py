"""eSCL (AirScan) scanner discovery and scanning over WiFi."""

from __future__ import annotations

import io
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests
from lxml import etree
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

NS_SCAN = "http://schemas.hp.com/imaging/escl/2011/05/03"
NS_PWG = "http://www.pwg.org/schemas/2010/12/sm"

SCAN_SETTINGS_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<scan:ScanSettings xmlns:scan="{ns_scan}" xmlns:pwg="{ns_pwg}">
  <pwg:Version>{version}</pwg:Version>
  <pwg:ScanRegions>
    <pwg:ScanRegion>
      <pwg:XOffset>0</pwg:XOffset>
      <pwg:YOffset>0</pwg:YOffset>
      <pwg:Width>{width}</pwg:Width>
      <pwg:Height>{height}</pwg:Height>
      <pwg:ContentRegionUnits>escl:ThreeHundredthsOfInches</pwg:ContentRegionUnits>
    </pwg:ScanRegion>
  </pwg:ScanRegions>
  <scan:InputSource>Platen</scan:InputSource>
  <pwg:DocumentFormat>{document_format}</pwg:DocumentFormat>
  <scan:ColorMode>{color_mode}</scan:ColorMode>
  <scan:XResolution>{x_resolution}</scan:XResolution>
  <scan:YResolution>{y_resolution}</scan:YResolution>
</scan:ScanSettings>"""

MAX_POLL_ATTEMPTS = 60
POLL_INTERVAL_SECONDS = 2
RETRY_ON_503_ATTEMPTS = 10
RETRY_ON_503_PAUSE = 1.0


@dataclass
class ScannerInfo:
    """A discovered eSCL scanner on the network."""

    name: str
    host: str
    port: int
    path: str
    make_and_model: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/{self.path.strip('/')}/"


@dataclass
class ScannerCapabilities:
    version: str = "2.0"
    make_and_model: str = ""
    formats: list[str] = field(default_factory=list)
    color_modes: list[str] = field(default_factory=list)
    x_resolutions: list[int] = field(default_factory=list)
    y_resolutions: list[int] = field(default_factory=list)
    max_width: int = 0
    max_height: int = 0


class ScannerError(Exception):
    pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class _Listener:
    """Collects discovered eSCL services."""

    def __init__(self) -> None:
        self.scanners: list[ScannerInfo] = []

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info: ServiceInfo | None = zc.get_service_info(type_, name)
        if info is None:
            return

        addresses = info.parsed_scoped_addresses()
        if not addresses:
            return

        host = addresses[0]
        port = info.port or 443
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in info.properties.items()
        }
        path = props.get("rs", "eSCL")

        self.scanners.append(
            ScannerInfo(
                name=name,
                host=host,
                port=port,
                path=path,
            )
        )

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass


def discover_scanners(timeout: float = 5.0) -> list[ScannerInfo]:
    """Discover eSCL scanners on the local network via mDNS.

    Args:
        timeout: How many seconds to wait for scanner announcements.

    Returns:
        A list of discovered scanners.
    """
    zc = Zeroconf()
    listener = _Listener()
    browser = ServiceBrowser(zc, "_uscan._tcp.local.", listener)
    time.sleep(timeout)
    browser.cancel()
    zc.close()
    return listener.scanners


# ---------------------------------------------------------------------------
# eSCL protocol helpers
# ---------------------------------------------------------------------------

def _xpath_text(tree: etree._Element, xpath: str) -> str | None:
    results = tree.xpath(xpath, namespaces={"pwg": NS_PWG, "scan": NS_SCAN})
    if results:
        return str(results[0])
    return None


def _xpath_texts(tree: etree._Element, xpath: str) -> list[str]:
    return [
        str(r)
        for r in tree.xpath(xpath, namespaces={"pwg": NS_PWG, "scan": NS_SCAN})
    ]


def _xpath_ints(tree: etree._Element, xpath: str) -> list[int]:
    return [
        int(r)
        for r in tree.xpath(xpath, namespaces={"pwg": NS_PWG, "scan": NS_SCAN})
    ]


def get_capabilities(base_url: str) -> ScannerCapabilities:
    """Query the scanner's eSCL capabilities."""
    url = urljoin(base_url, "ScannerCapabilities")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    tree = etree.fromstring(resp.content)

    caps = ScannerCapabilities()
    caps.version = _xpath_text(tree, "//pwg:Version/text()") or "2.0"
    caps.make_and_model = _xpath_text(tree, "//pwg:MakeAndModel/text()") or ""
    caps.formats = _xpath_texts(tree, "//pwg:DocumentFormat/text()")
    caps.color_modes = _xpath_texts(tree, "//scan:ColorMode/text()")
    caps.x_resolutions = _xpath_ints(
        tree, "//scan:PlatenInputCaps//scan:DiscreteResolutions//scan:DiscreteResolution/scan:XResolution/text()"
    )
    caps.y_resolutions = _xpath_ints(
        tree, "//scan:PlatenInputCaps//scan:DiscreteResolutions//scan:DiscreteResolution/scan:YResolution/text()"
    )
    # Fallback: some scanners list resolutions outside PlatenInputCaps
    if not caps.x_resolutions:
        caps.x_resolutions = _xpath_ints(tree, "//scan:XResolution/text()")
    if not caps.y_resolutions:
        caps.y_resolutions = _xpath_ints(tree, "//scan:YResolution/text()")

    max_w = _xpath_text(tree, "//scan:PlatenInputCaps/scan:MaxWidth/text()")
    max_h = _xpath_text(tree, "//scan:PlatenInputCaps/scan:MaxHeight/text()")
    if not max_w:
        max_w = _xpath_text(tree, "//scan:MaxWidth/text()")
    if not max_h:
        max_h = _xpath_text(tree, "//scan:MaxHeight/text()")
    caps.max_width = int(max_w) if max_w else 2550
    caps.max_height = int(max_h) if max_h else 3508

    return caps


def get_status(base_url: str) -> str:
    """Return the scanner's current state string (e.g. 'Idle', 'Processing')."""
    url = urljoin(base_url, "ScannerStatus")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    tree = etree.fromstring(resp.content)
    state = _xpath_text(tree, "//pwg:State/text()")
    return state or "Unknown"


def _pick_resolution(caps: ScannerCapabilities, requested_dpi: int) -> int:
    """Pick the best available resolution closest to the requested DPI."""
    available = sorted(set(caps.x_resolutions) & set(caps.y_resolutions))
    if not available:
        return requested_dpi
    if requested_dpi in available:
        return requested_dpi
    return min(available, key=lambda r: abs(r - requested_dpi))


def scan(
    base_url: str,
    dpi: int = 300,
    color_mode: str = "RGB24",
    document_format: str = "image/jpeg",
    on_status: None | (callable) = None,
) -> bytes:
    """Execute a scan and return the image bytes.

    Args:
        base_url: The scanner's eSCL base URL (e.g. http://192.168.1.5:80/eSCL/).
        dpi: Requested scan resolution.
        color_mode: eSCL color mode (RGB24, Grayscale8).
        document_format: MIME type for the scan output.
        on_status: Optional callback receiving status strings for progress reporting.

    Returns:
        Raw image bytes of the scanned image.

    Raises:
        ScannerError: If the scanner is not idle or the scan fails.
    """
    def _report(msg: str) -> None:
        if on_status:
            on_status(msg)

    caps = get_capabilities(base_url)
    _report(f"Scanner: {caps.make_and_model or 'Unknown'}")

    resolution = _pick_resolution(caps, dpi)
    if resolution != dpi:
        _report(f"Requested {dpi} DPI not available, using {resolution} DPI")

    if document_format not in caps.formats:
        fallback = "image/jpeg" if "image/jpeg" in caps.formats else caps.formats[0] if caps.formats else document_format
        if fallback != document_format:
            _report(f"Format {document_format} not supported, using {fallback}")
            document_format = fallback

    if color_mode not in caps.color_modes:
        fallback = "RGB24" if "RGB24" in caps.color_modes else caps.color_modes[0] if caps.color_modes else color_mode
        if fallback != color_mode:
            _report(f"Color mode {color_mode} not supported, using {fallback}")
            color_mode = fallback

    status = get_status(base_url)
    if status != "Idle":
        raise ScannerError(f"Scanner is not idle (state: {status}). Please wait and try again.")

    _report("Starting scan...")
    settings_xml = SCAN_SETTINGS_TEMPLATE.format(
        ns_scan=NS_SCAN,
        ns_pwg=NS_PWG,
        version=caps.version,
        width=caps.max_width,
        height=caps.max_height,
        document_format=document_format,
        color_mode=color_mode,
        x_resolution=resolution,
        y_resolution=resolution,
    )

    jobs_url = urljoin(base_url, "ScanJobs")
    resp = requests.post(
        jobs_url,
        data=settings_xml,
        headers={"Content-Type": "text/xml"},
        timeout=30,
        allow_redirects=False,
    )

    if resp.status_code not in (200, 201):
        raise ScannerError(f"Failed to create scan job: HTTP {resp.status_code} - {resp.text}")

    location = resp.headers.get("Location", "")
    if not location:
        raise ScannerError("Scanner did not return a Location header for the scan job.")

    document_url = urljoin(location + "/", "NextDocument")
    _report(f"Scan job created, waiting for result...")

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            doc_resp = requests.get(document_url, timeout=60)
        except requests.RequestException as exc:
            _report(f"Poll attempt {attempt}: connection error ({exc}), retrying...")
            continue

        if doc_resp.status_code == 200:
            _report(f"Scan complete ({len(doc_resp.content)} bytes)")
            return doc_resp.content

        if doc_resp.status_code == 503:
            _report(f"Scanner busy (503), retrying... ({attempt}/{MAX_POLL_ATTEMPTS})")
            continue

        if doc_resp.status_code == 404:
            _report(f"Not ready yet (404), retrying... ({attempt}/{MAX_POLL_ATTEMPTS})")
            continue

        raise ScannerError(f"Unexpected response while fetching scan: HTTP {doc_resp.status_code}")

    raise ScannerError(f"Timed out waiting for scan after {MAX_POLL_ATTEMPTS} attempts.")
