import glob
import os
import pty
import re
import sys
import termios
import asyncio
import subprocess
from typing import AsyncGenerator, Optional

FIRMWARE_REPO_URL = "https://github.com/MeterMonitor-io/MeterMonitor-esp"
FIRMWARE_DIR = "/data/metermonitor-esp"


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
            inner = raw[1:-1]
            # `default "y"` / `default "n"` are boolean, not strings
            if inner.lower() == "y":
                return True
            if inner.lower() == "n":
                return False
            return inner
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


def get_kconfig_options(target_menus: list = None) -> list:
    """
    Collect options from every Kconfig file that contains any of the target menus.
    main/Kconfig.projbuild is checked first (it's the standard home for
    project-level ESP-IDF configuration such as WiFi / MQTT / camera settings).
    Options from later files/menus only fill in names not yet seen.
    """
    if target_menus is None:
        target_menus = ["MeterMonitor", "Example Connection Configuration"]

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
        for menu in target_menus:
            opts = _parse_kconfig_content(content, menu)
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
    # Mirror MeterMonitor WiFi options into the ESP-IDF example connection component
    # keys so that CONFIG_EXAMPLE_WIFI_SSID / _PASSWORD are compiled in correctly.
    # The firmware's WiFi fallback reads those Kconfig symbols when NVS has no config.
    merged = dict(config_values)
    if "METER_MONITOR_WIFI_SSID" in merged and "EXAMPLE_WIFI_SSID" not in merged:
        merged["EXAMPLE_WIFI_SSID"] = merged["METER_MONITOR_WIFI_SSID"]
    if "METER_MONITOR_WIFI_PASSWORD" in merged and "EXAMPLE_WIFI_PASSWORD" not in merged:
        merged["EXAMPLE_WIFI_PASSWORD"] = merged["METER_MONITOR_WIFI_PASSWORD"]
    config_values = merged

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


def _find_idf_path() -> str:
    candidates = [
        os.environ.get("IDF_PATH", ""),
        "/opt/esp-idf",
        os.path.expanduser("~/esp/esp-idf"),
        os.path.expanduser("~/.espressif/esp-idf"),
    ]
    return next(
        (p for p in candidates if p and os.path.isfile(os.path.join(p, "export.sh"))),
        "",
    )


async def _get_idf_build_env(idf_path: str) -> dict:
    """
    Build the full environment needed to run idf.py build.

    1. Run idf_tools.py export (standalone script, any Python) for the official
       KEY=VALUE pairs — PATH, OPENOCD_SCRIPTS, etc.
    2. Supplement with a direct glob over ~/.espressif/tools so the xtensa
       compiler is guaranteed to be in PATH even if idf_tools.py produces no
       output (e.g. partial Docker environment).
    3. Explicitly set IDF_PYTHON_ENV_PATH so idf.py doesn't warn and internally
       reconstruct the environment incorrectly.
    """
    espressif_dir = os.path.expanduser("~/.espressif")
    tools_dir = os.path.join(espressif_dir, "tools")

    env = {
        **os.environ,
        "IDF_PATH": idf_path,
        "IDF_TOOLS_PATH": espressif_dir,
    }

    # ── 1. idf_tools.py export ─────────────────────────────────────────────
    idf_tools_py = os.path.join(idf_path, "tools", "idf_tools.py")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, idf_tools_py, "export", "--format=key-value",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if key == "PATH":
                val = val.replace("$PATH", os.environ.get("PATH", ""))
            env[key] = val
    except Exception:
        pass

    # ── 2. Glob-based PATH supplement ──────────────────────────────────────
    # Guarantees the xtensa compiler is reachable even when idf_tools.py
    # produces partial or no output (common in minimal Docker environments).
    tool_patterns = [
        "xtensa-esp-elf/*/xtensa-esp-elf/bin",   # ESP-IDF 5.x naming
        "xtensa-esp32-elf/*/bin",                  # older naming
        "esp32ulp-elf/*/esp32ulp-elf/bin",
        "cmake/*/bin",
        "ninja/*/",
    ]
    extra: list[str] = []
    for pat in tool_patterns:
        extra.extend(sorted(glob.glob(os.path.join(tools_dir, pat))))
    extra.append(os.path.join(idf_path, "tools"))

    current_path = env.get("PATH", os.environ.get("PATH", ""))
    new_dirs = [p for p in extra if os.path.isdir(p) and p not in current_path]
    if new_dirs:
        env["PATH"] = os.pathsep.join(new_dirs) + os.pathsep + current_path

    # ── 3. IDF_PYTHON_ENV_PATH ─────────────────────────────────────────────
    if "IDF_PYTHON_ENV_PATH" not in env:
        venv_dirs = sorted(
            glob.glob(os.path.join(espressif_dir, "python_env", "idf*_env"))
        )
        if venv_dirs:
            env["IDF_PYTHON_ENV_PATH"] = venv_dirs[-1]

    return env


def _find_idf_venv_python(env: dict) -> str:
    """Return the Python executable from the ESP-IDF virtualenv."""
    venv_root = env.get("IDF_PYTHON_ENV_PATH", "")
    for candidate in (
        os.path.join(venv_root, "bin", "python"),
        os.path.join(venv_root, "bin", "python3"),
    ):
        if venv_root and os.path.isfile(candidate):
            return candidate

    # Fallback: glob
    matches = sorted(
        glob.glob(os.path.expanduser("~/.espressif/python_env/idf*_env/bin/python"))
    )
    return matches[-1] if matches else sys.executable


def _apply_firmware_source_patches() -> None:
    """Patch known bugs in the firmware source before building."""
    nvs_helper = os.path.join(FIRMWARE_DIR, "main", "nvs_helper.c")
    if not os.path.isfile(nvs_helper):
        return
    with open(nvs_helper) as f:
        src = f.read()
    if "strncpy" in src and "<string.h>" not in src:
        # Insert #include <string.h> after the last existing #include line
        lines = src.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                insert_at = i + 1
        lines.insert(insert_at, "#include <string.h>\n")
        with open(nvs_helper, "w") as f:
            f.writelines(lines)


def _apply_idf_cmake_workarounds(idf_path: str) -> None:
    """
    Patch known ESP-IDF cmake bugs that surface at build time, not at image-build time.

    1. gdbinit.cmake iterates git submodules with an unquoted cmake variable:
           file(TO_CMAKE_PATH ${dir} result)
       When `git submodule foreach` returns nothing (non-submodule git init),
       ${dir} expands to zero tokens → CMake error "must be called with exactly
       three arguments."  The file is only needed for interactive GDB sessions,
       never for building, so we replace it with a no-op stub.

    2. GetGitRevisionDescription writes a head-ref file by running
           git rev-parse HEAD
       in $IDF_PATH.  If that repo has no commits (bare `git init` without a
       commit), HEAD is unresolvable, the file is never written, and the
       subsequent include() fails.  We create a minimal tagged empty commit so
       rev-parse succeeds.
    """
    # ── 1. gdbinit.cmake stub ──────────────────────────────────────────────
    gdbinit = os.path.join(idf_path, "tools", "cmake", "gdbinit.cmake")
    if os.path.isfile(gdbinit):
        with open(gdbinit) as f:
            content = f.read()
        if "TO_CMAKE_PATH ${dir}" in content or "submodule foreach" in content:
            with open(gdbinit, "w") as f:
                f.write("function(__generate_gdbinit)\nendfunction()\n")

    # ── 2. Ensure IDF git repo has a HEAD-resolvable commit ───────────────
    git_dir = os.path.join(idf_path, ".git")
    if not os.path.isdir(git_dir):
        return
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=idf_path,
        capture_output=True,
    )
    if result.returncode == 0:
        return  # already fine
    # No valid HEAD — create a minimal empty commit so cmake can resolve it.
    for cmd in (
        ["git", "-C", idf_path, "init", "-q"],
        ["git", "-C", idf_path, "-c", "user.email=b@b", "-c", "user.name=b",
         "commit", "--allow-empty", "-q", "-m", "esp-idf"],
    ):
        subprocess.run(cmd, capture_output=True)


async def build_firmware_stream(config_values: dict) -> AsyncGenerator[str, None]:
    apply_sdkconfig(config_values)

    idf_path = _find_idf_path()
    if not idf_path:
        yield (
            "ERROR: ESP-IDF not found.\n"
            "Set the IDF_PATH environment variable or install ESP-IDF to ~/esp/esp-idf.\n"
        )
        yield "__BUILD_FAILED__1__\n"
        return

    _apply_idf_cmake_workarounds(idf_path)
    _apply_firmware_source_patches()

    idf_py = os.path.join(idf_path, "tools", "idf.py")

    env = await _get_idf_build_env(idf_path)
    env["PYTHONUNBUFFERED"] = "1"
    venv_python = _find_idf_venv_python(env)

    # Use a pty instead of a plain pipe so that idf.py and ninja see a terminal
    # and switch from fully-buffered to line-buffered output, giving us live
    # progress lines instead of everything arriving in one chunk at the end.
    master_fd, slave_fd = pty.openpty()
    try:
        # Disable NL→CRNL conversion so output contains plain \n, not \r\n
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] &= ~termios.ONLCR
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
    except Exception:
        pass

    proc = None
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _on_readable():
        try:
            data = os.read(master_fd, 4096)
            if data:
                queue.put_nowait(data)
        except OSError:
            loop.remove_reader(master_fd)
            queue.put_nowait(None)  # EOF sentinel

    try:
        proc = await asyncio.create_subprocess_exec(
            venv_python, idf_py, "build",
            cwd=FIRMWARE_DIR,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)
        slave_fd = -1
        loop.add_reader(master_fd, _on_readable)

        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                # Heartbeat: prevents the HA ingress proxy from closing the
                # connection during long compilation steps with no output.
                yield "\n"
                continue
            if data is None:
                break
            yield data.decode("utf-8", errors="replace")

        await proc.wait()
        if proc.returncode == 0:
            yield "__BUILD_SUCCESS__\n"
        else:
            yield f"__BUILD_FAILED__{proc.returncode}__\n"
    except Exception as e:
        yield f"ERROR: {e}\n"
        yield "__BUILD_FAILED__1__\n"
    finally:
        try:
            loop.remove_reader(master_fd)
        except Exception:
            pass
        if slave_fd != -1:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


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
