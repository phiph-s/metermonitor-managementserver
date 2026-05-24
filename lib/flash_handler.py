import os
import re
import asyncio
import subprocess
from typing import AsyncGenerator, Optional

FIRMWARE_REPO_URL = "https://github.com/MeterMonitor-io/MeterMonitor-esp"
FIRMWARE_DIR = "/tmp/metermonitor-esp"


def clone_or_pull_firmware() -> dict:
    git_dir = os.path.join(FIRMWARE_DIR, ".git")
    if os.path.isdir(git_dir):
        result = subprocess.run(
            ["git", "pull", "--recurse-submodules"],
            cwd=FIRMWARE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "action": "pull",
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
        }
    else:
        os.makedirs(FIRMWARE_DIR, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--recurse-submodules", FIRMWARE_REPO_URL, FIRMWARE_DIR],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "action": "clone",
            "success": result.returncode == 0,
            "output": result.stdout + result.stderr,
        }


def firmware_exists() -> bool:
    return os.path.isdir(os.path.join(FIRMWARE_DIR, ".git"))


def _find_kconfig_files(project_dir: str) -> list:
    skip_dirs = {"build", ".git", "managed_components", "__pycache__"}
    result = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname in ("Kconfig", "Kconfig.projbuild"):
                result.append(os.path.join(root, fname))
    return result


def _parse_kconfig_content(content: str, target_menu: str) -> Optional[list]:
    lines = content.splitlines()

    menu_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(
            r'menu\s+"[^"]*' + re.escape(target_menu) + r'[^"]*"',
            stripped,
            re.IGNORECASE,
        ):
            menu_start = i
            break

    if menu_start == -1:
        return None

    options: list = []
    depth = 1
    current: Optional[dict] = None       # current regular config
    in_choice: Optional[dict] = None     # current choice block
    choice_item: Optional[dict] = None   # current config inside choice
    in_help = False
    help_indent = 0
    i = menu_start + 1

    def flush_regular():
        nonlocal current
        if current:
            # Skip internal options with no user-visible prompt
            if current["label"] is None:
                current = None
                return
            current["help"] = current["help"].strip()
            options.append(current)
            current = None

    def flush_choice_item():
        nonlocal choice_item
        if choice_item and in_choice is not None:
            in_choice["choices"].append(
                {
                    "name": choice_item["name"],
                    "label": choice_item["label"],
                    "help": choice_item["help"].strip(),
                }
            )
            choice_item = None

    def flush_choice():
        nonlocal in_choice
        flush_choice_item()
        if in_choice:
            in_choice["help"] = in_choice["help"].strip()
            options.append(in_choice)
            in_choice = None

    def _help_target() -> Optional[dict]:
        return choice_item or current or in_choice

    def _parse_default(raw: str):
        raw = raw.strip().split("#")[0].strip()
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1]
        if raw.lower() == "y":
            return True
        if raw.lower() == "n":
            return False
        return raw

    while i < len(lines) and depth > 0:
        line = lines[i]
        stripped = line.strip()
        # Expand tabs before computing indent to handle mixed tab/space indentation
        _expanded = line.expandtabs(4)
        indent = len(_expanded) - len(_expanded.lstrip())

        if stripped.startswith("menu ") and not stripped.startswith("menuconfig"):
            depth += 1
        elif stripped == "endmenu":
            depth -= 1
            if depth == 0:
                break

        if depth >= 1:
            # ── help text collection ───────────────────────────────────────
            if in_help:
                tgt = _help_target()
                if not stripped:
                    if tgt:
                        tgt["help"] += "\n"
                elif indent > help_indent:
                    if tgt:
                        tgt["help"] += stripped + " "
                else:
                    in_help = False
                    # fall through to re-process this line

            if not in_help:
                # ── choice start ───────────────────────────────────────────
                if re.match(r"^choice\b", stripped):
                    flush_regular()
                    parts = stripped.split()
                    cname = parts[1] if len(parts) > 1 else f"__CHOICE_{len(options)}__"
                    in_choice = {
                        "name": cname,
                        "type": "choice",
                        "label": cname.replace("_", " ").title(),
                        "default": None,
                        "help": "",
                        "depends_on": None,
                        "choices": [],
                    }

                elif stripped == "endchoice":
                    flush_choice()

                elif in_choice is not None:
                    # ── inside choice block ────────────────────────────────
                    m = re.match(r'prompt\s+"([^"]+)"', stripped)
                    if m:
                        in_choice["label"] = m.group(1)

                    # Accept `bool/string/int/hex "label"` as the choice's prompt
                    m = re.match(r'(?:bool|int|hex|string)\s+"([^"]+)"', stripped)
                    if m and choice_item is None:
                        in_choice["label"] = m.group(1)

                    m = re.match(r"default\s+(\S.*)", stripped)
                    if m and not stripped.startswith("config "):
                        in_choice["default"] = _parse_default(m.group(1))

                    m = re.match(r"depends\s+on\s+(.*)", stripped)
                    if m:
                        in_choice["depends_on"] = m.group(1).strip()

                    if stripped.startswith("config "):
                        flush_choice_item()
                        name = stripped.split(None, 1)[1].strip()
                        choice_item = {
                            "name": name,
                            "label": name.replace("_", " ").title(),
                            "help": "",
                        }
                    elif choice_item is not None:
                        m = re.match(r'bool\s+"([^"]+)"', stripped)
                        if m:
                            choice_item["label"] = m.group(1)
                        if stripped in ("help", "---help---"):
                            in_help = True
                            help_indent = indent
                    elif stripped in ("help", "---help---"):
                        # help text for the choice block itself
                        in_help = True
                        help_indent = indent

                else:
                    # ── regular config ─────────────────────────────────────
                    if stripped.startswith("config "):
                        flush_regular()
                        name = stripped.split(None, 1)[1].strip()
                        current = {
                            "name": name,
                            "type": "string",
                            "label": None,   # None until an explicit prompt is found
                            "default": None,
                            "help": "",
                            "depends_on": None,
                            "range": None,
                        }
                    elif current is not None:
                        for typ in ("bool", "string", "int", "hex"):
                            m = re.match(rf"^{typ}(?:\s+\"([^\"]+)\")?$", stripped)
                            if m:
                                current["type"] = typ
                                if m.group(1):
                                    current["label"] = m.group(1)
                                break

                        m = re.match(r'prompt\s+"([^"]+)"', stripped)
                        if m:
                            current["label"] = m.group(1)

                        m = re.match(r"default\s+(.*)", stripped)
                        if m:
                            current["default"] = _parse_default(m.group(1))

                        m = re.match(r"depends\s+on\s+(.*)", stripped)
                        if m:
                            current["depends_on"] = m.group(1).strip()

                        m = re.match(r"range\s+(\S+)\s+(\S+)", stripped)
                        if m:
                            current["range"] = [m.group(1), m.group(2)]

                        if stripped in ("help", "---help---"):
                            in_help = True
                            help_indent = indent

        i += 1

    flush_regular()
    flush_choice()
    return options


def get_kconfig_options(target_menu: str = "MeterMonitor") -> list:
    """
    Collect options from every Kconfig file that contains the target menu.
    main/Kconfig.projbuild is checked first (it's the standard home for
    project-level ESP-IDF configuration such as WiFi / MQTT / camera settings).
    Options from later files only fill in names not yet seen.
    """
    seen: set = set()
    merged: list = []

    # Build ordered list: main/Kconfig.projbuild first, then everything else
    primary = os.path.join(FIRMWARE_DIR, "main", "Kconfig.projbuild")
    ordered = []
    if os.path.isfile(primary):
        ordered.append(primary)
    for kpath in _find_kconfig_files(FIRMWARE_DIR):
        if kpath != primary:
            ordered.append(kpath)

    for kpath in ordered:
        try:
            with open(kpath) as f:
                content = f.read()
        except OSError:
            continue
        opts = _parse_kconfig_content(content, target_menu)
        if opts is None:
            continue
        for opt in opts:
            if opt["name"] not in seen:
                seen.add(opt["name"])
                merged.append(opt)

    return merged


def read_current_sdkconfig() -> dict:
    values: dict = {}
    for fname in ("sdkconfig.defaults", "sdkconfig"):
        fpath = os.path.join(FIRMWARE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("CONFIG_") and "=" in line:
                    key, val = line.split("=", 1)
                    name = key[7:]
                    if val == "y":
                        values[name] = True
                    elif val == "n":
                        values[name] = False
                    elif val.startswith('"') and val.endswith('"'):
                        values[name] = val[1:-1]
                    else:
                        values[name] = val
                elif "# CONFIG_" in line and " is not set" in line:
                    m = re.search(r"# (CONFIG_\S+) is not set", line)
                    if m:
                        values[m.group(1)[7:]] = False
    return values


def apply_sdkconfig(config_values: dict):
    """Write user values into sdkconfig.defaults, and patch sdkconfig if it exists."""
    defaults_path = os.path.join(FIRMWARE_DIR, "sdkconfig.defaults")
    managed_keys = {f"CONFIG_{n}" for n in config_values}

    existing: list = []
    if os.path.exists(defaults_path):
        with open(defaults_path) as f:
            existing = f.readlines()

    kept = []
    for line in existing:
        s = line.strip()
        skip = False
        for mk in managed_keys:
            if s.startswith(mk + "=") or s == f"# {mk} is not set":
                skip = True
                break
        if not skip:
            kept.append(line)

    for name, value in config_values.items():
        key = f"CONFIG_{name}"
        if isinstance(value, bool):
            kept.append(f"{key}={'y' if value else 'n'}\n")
        elif isinstance(value, (int, float)):
            kept.append(f"{key}={value}\n")
        else:
            kept.append(f'{key}="{value}"\n')

    with open(defaults_path, "w") as f:
        f.writelines(kept)

    sdkconfig_path = os.path.join(FIRMWARE_DIR, "sdkconfig")
    if not os.path.exists(sdkconfig_path):
        return

    with open(sdkconfig_path) as f:
        sc_lines = f.readlines()

    new_sc: list = []
    patched: set = set()
    for line in sc_lines:
        s = line.strip()
        key = None
        if s.startswith("CONFIG_") and "=" in s:
            key = s.split("=")[0]
        elif "# CONFIG_" in s and " is not set" in s:
            m = re.search(r"# (CONFIG_\S+) is not set", s)
            if m:
                key = m.group(1)

        if key and key in managed_keys:
            if key not in patched:
                patched.add(key)
                name = key[7:]
                v = config_values[name]
                if isinstance(v, bool):
                    new_sc.append(f"{key}={'y' if v else 'n'}\n")
                elif isinstance(v, (int, float)):
                    new_sc.append(f"{key}={v}\n")
                else:
                    new_sc.append(f'{key}="{v}"\n')
        else:
            new_sc.append(line)

    for name, v in config_values.items():
        key = f"CONFIG_{name}"
        if key not in patched:
            if isinstance(v, bool):
                new_sc.append(f"{key}={'y' if v else 'n'}\n")
            elif isinstance(v, (int, float)):
                new_sc.append(f"{key}={v}\n")
            else:
                new_sc.append(f'{key}="{v}"\n')

    with open(sdkconfig_path, "w") as f:
        f.writelines(new_sc)


async def _get_idf_env() -> tuple[dict, str]:
    """
    Return (env_dict, idf_py_path) with the full ESP-IDF environment sourced.
    Looks for IDF_PATH in the process environment or common install locations.
    Sources export.sh so the xtensa toolchain ends up in PATH.
    """
    candidates = [
        os.environ.get("IDF_PATH", ""),
        "/opt/esp-idf",
        os.path.expanduser("~/esp/esp-idf"),
        os.path.expanduser("~/.espressif/esp-idf"),
    ]

    idf_path = next((p for p in candidates if p and os.path.isfile(os.path.join(p, "export.sh"))), "")

    if not idf_path:
        # Nothing found; return bare environment and let idf.py fail with a clear message
        return os.environ.copy(), "idf.py"

    export_sh = os.path.join(idf_path, "export.sh")

    # Source export.sh in a bash subprocess and capture the resulting environment
    proc = await asyncio.create_subprocess_shell(
        f'. "{export_sh}" > /dev/null 2>&1 && env -0',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        executable="/bin/bash",
    )
    stdout, _ = await proc.communicate()

    env: dict = {}
    for entry in stdout.decode("utf-8", errors="replace").split("\0"):
        if "=" in entry:
            key, _, val = entry.partition("=")
            env[key] = val

    # Fallback: merge with current env so nothing critical is missing
    merged = {**os.environ, **env}

    idf_py = os.path.join(idf_path, "tools", "idf.py")
    if not os.path.isfile(idf_py):
        idf_py = "idf.py"

    return merged, idf_py


async def build_firmware_stream(config_values: dict) -> AsyncGenerator[str, None]:
    apply_sdkconfig(config_values)

    env, idf_cmd = await _get_idf_env()

    if idf_cmd == "idf.py" and not os.path.isfile(idf_cmd):
        yield (
            "ERROR: ESP-IDF not found.\n"
            "Set the IDF_PATH environment variable to your ESP-IDF installation directory,\n"
            "or install it to ~/esp/esp-idf (see the documentation).\n"
        )
        yield "__BUILD_FAILED__1__\n"
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            idf_cmd,
            "build",
            cwd=FIRMWARE_DIR,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        async for raw in proc.stdout:
            yield raw.decode("utf-8", errors="replace")
        await proc.wait()
        if proc.returncode == 0:
            yield "__BUILD_SUCCESS__\n"
        else:
            yield f"__BUILD_FAILED__{proc.returncode}__\n"
    except FileNotFoundError:
        yield "ERROR: idf.py not found. Ensure ESP-IDF is installed and IDF_PATH is set.\n"
        yield "__BUILD_FAILED__1__\n"


def get_flash_args() -> dict:
    flash_args_path = os.path.join(FIRMWARE_DIR, "build", "flash_args")
    if not os.path.exists(flash_args_path):
        raise FileNotFoundError("flash_args not found – build the firmware first.")

    with open(flash_args_path) as f:
        content = f.read().strip()

    args: dict = {
        "flash_mode": "dio",
        "flash_freq": "40m",
        "flash_size": "detect",
        "binaries": [],
    }

    for line in content.splitlines():
        line = line.strip()
        m = re.search(r"--flash_mode\s+(\S+)", line)
        if m:
            args["flash_mode"] = m.group(1)
        m = re.search(r"--flash_freq\s+(\S+)", line)
        if m:
            args["flash_freq"] = m.group(1)
        m = re.search(r"--flash_size\s+(\S+)", line)
        if m:
            args["flash_size"] = m.group(1)
        m = re.match(r"(0x[0-9a-fA-F]+)\s+(\S+\.bin)", line)
        if m:
            offset = int(m.group(1), 16)
            rel_path = m.group(2)
            bin_path = os.path.join(FIRMWARE_DIR, "build", rel_path)
            if os.path.exists(bin_path):
                args["binaries"].append(
                    {
                        "offset": offset,
                        "path": rel_path,
                        "filename": os.path.basename(rel_path),
                        "size": os.path.getsize(bin_path),
                    }
                )

    return args
