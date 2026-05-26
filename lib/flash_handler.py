import json
import os
import tempfile
import zipfile
import urllib.request

FIRMWARE_CACHE_DIR = "/data/firmware-cache"
RELEASES_REPO      = "MeterMonitor-io/metermonitor-firmware-mqtt"
NVS_PARTITION_OFFSET = 0x9000
NVS_PARTITION_SIZE   = 0x6000   # must match partitions.csv in firmware repo

BOARDS = ["ai_thinker", "m5cam_a", "m5cam_b", "esp_eye", "esp32s3_eye"]

BOARD_LABELS = {
    "ai_thinker":  "AI-Thinker / Freenove / TTGO T-Journal",
    "m5cam_a":     "M5-Camera Model A",
    "m5cam_b":     "M5-Camera Model B (PSRAM)",
    "esp_eye":     "ESP-EYE / TTGO T-Camera Plus",
    "esp32s3_eye": "ESP32-S3-EYE",
}


# ── Releases ───────────────────────────────────────────────────────────────────

def get_releases() -> list:
    url = f"https://api.github.com/repos/{RELEASES_REPO}/releases"
    req = urllib.request.Request(url, headers={
        "Accept":     "application/vnd.github+json",
        "User-Agent": "metermonitor",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            releases = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Failed to fetch releases: {e}")

    result = []
    for rel in releases:
        tag    = rel["tag_name"]
        assets = {a["name"] for a in rel.get("assets", [])}
        boards = [b for b in BOARDS if f"metermonitor-{b}.zip" in assets]
        if boards:
            result.append({
                "tag":    tag,
                "name":   rel.get("name") or tag,
                "boards": boards,
            })
    return result


# ── Download ───────────────────────────────────────────────────────────────────

def _cache_dir(tag: str, board: str) -> str:
    return os.path.join(FIRMWARE_CACHE_DIR, tag, board)


def download_release(tag: str, board: str) -> None:
    if board not in BOARDS:
        raise ValueError(f"Unknown board: {board!r}")

    dest   = _cache_dir(tag, board)
    marker = os.path.join(dest, ".downloaded")
    if os.path.exists(marker):
        return  # already cached

    os.makedirs(dest, exist_ok=True)
    url = (
        f"https://github.com/{RELEASES_REPO}/releases/download/"
        f"{tag}/metermonitor-{board}.zip"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "metermonitor"})
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp_path, "wb") as f:
                f.write(resp.read())
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dest)
        open(marker, "w").close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def is_downloaded(tag: str, board: str) -> bool:
    return os.path.exists(os.path.join(_cache_dir(tag, board), ".downloaded"))


# ── NVS binary generation ──────────────────────────────────────────────────────

def generate_nvs_binary(config: dict) -> bytes:
    from esp_idf_nvs_partition_gen import nvs_partition_gen

    rows = [
        "key,type,encoding,value",
        "config,namespace,,",
    ]
    fields = [
        ("wifi_ssid",  "string", config.get("wifi_ssid",  "")),
        ("wifi_pass",  "string", config.get("wifi_pass",  "")),
        ("mqtt_url",   "string", config.get("mqtt_url",   "mqtt://192.168.1.1:1883")),
        ("mqtt_user",  "string", config.get("mqtt_user",  "")),
        ("mqtt_pass",  "string", config.get("mqtt_pass",  "")),
        ("mqtt_topic", "string", config.get("mqtt_topic", "MeterMonitor/meter")),
        ("meter_name", "string", config.get("meter_name", "meter")),
        ("interval",   "u32",    str(int(config.get("interval", 30)))),
        ("flash_en",   "u8",     "1" if config.get("flash_en", True) else "0"),
    ]
    for key, encoding, value in fields:
        safe = str(value).replace('"', '""')
        rows.append(f"{key},data,{encoding},{safe}")

    csv_text = "\n".join(rows) + "\n"

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "nvs.csv")
        bin_path = os.path.join(d, "nvs.bin")
        with open(csv_path, "w") as f:
            f.write(csv_text)
        nvs_partition_gen.generate(
            nvs_partition_gen.parse_args([
                "generate", csv_path, bin_path, hex(NVS_PARTITION_SIZE),
            ])
        )
        with open(bin_path, "rb") as f:
            return f.read()


# ── Flash args ─────────────────────────────────────────────────────────────────

def get_flash_args(tag: str, board: str) -> dict:
    dest      = _cache_dir(tag, board)
    json_path = os.path.join(dest, "flasher_args.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "Firmware not yet downloaded — call POST /api/flash/download first."
        )

    with open(json_path) as f:
        data = json.load(f)

    settings = data.get("flash_settings", {})
    args: dict = {
        "flash_mode": settings.get("flash_mode", "dio"),
        "flash_freq": settings.get("flash_freq", "40m"),
        "flash_size": settings.get("flash_size", "detect"),
        "binaries":   [],
    }

    for hex_offset, rel_path in data.get("flash_files", {}).items():
        offset   = int(hex_offset, 16)
        filename = os.path.basename(rel_path)
        # zip -j flattened the archive, so files sit directly in dest
        bin_path = os.path.join(dest, filename)
        if not os.path.exists(bin_path):
            bin_path = os.path.join(dest, rel_path)
        if os.path.exists(bin_path):
            args["binaries"].append({
                "offset":   offset,
                "path":     f"{tag}/{board}/{filename}",
                "filename": filename,
                "size":     os.path.getsize(bin_path),
            })

    return args
