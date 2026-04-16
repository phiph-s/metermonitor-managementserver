from lib.log import log

import asyncio
import json
from typing import Optional, Dict, Any, List

import requests
import websockets


def _to_ws_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://"):] + "/api/websocket"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://"):] + "/api/websocket"
    return "ws://" + base_url + "/api/websocket"


async def _fetch_entity_registry(base_url: str, token: str) -> List[Dict[str, Any]]:
    ws_url = _to_ws_url(base_url)

    async with websockets.connect(ws_url) as ws:
        # hello
        await ws.recv()

        # auth
        await ws.send(json.dumps({
            "type": "auth",
            "access_token": token
        }))
        auth_resp = json.loads(await ws.recv())
        if auth_resp.get("type") != "auth_ok":
            raise RuntimeError(f"HA WS auth failed: {auth_resp}")

        # entity registry
        await ws.send(json.dumps({
            "id": 1,
            "type": "config/entity_registry/list"
        }))
        resp = json.loads(await ws.recv())
        if not resp.get("success"):
            raise RuntimeError(f"entity_registry/list failed: {resp}")

        return resp["result"]


def _score(entity_id: str, friendly_name: str) -> int:
    eid = (entity_id or "").lower()
    fn = (friendly_name or "").lower()

    score = 0
    if "flash" in eid or "flash" in fn:
        score += 10
    if "led" in eid or "led" in fn:
        score += 5
    return score


def suggest_flash_entity(
    ha_base_url: str,
    ha_token: str,
    camera_entity_id: str,
    states: list
) -> Optional[str]:
    """
    Returns a suggested light.* entity_id that belongs to the same device
    as the given camera entity.
    """

    try:
        ent_reg = asyncio.run(
            _fetch_entity_registry(ha_base_url, ha_token)
        )
    except Exception as e:
        log(f"[FLASH-SUGGEST] WS error: {e}")
        return None

    cam_entry = next(
        (e for e in ent_reg if e.get("entity_id") == camera_entity_id),
        None
    )
    if not cam_entry:
        return None

    device_id = cam_entry.get("device_id")
    if not device_id:
        return None

    # Map states for friendly_name lookup
    state_by_eid = {
        s.get("entity_id"): s
        for s in states
        if isinstance(s.get("entity_id"), str)
    }

    candidates = []
    for e in ent_reg:
        if e.get("device_id") != device_id:
            continue

        eid = e.get("entity_id")
        if isinstance(eid, str) and eid.startswith("light."):
            fn = (
                state_by_eid
                .get(eid, {})
                .get("attributes", {})
                .get("friendly_name", "")
            )
            candidates.append(( _score(eid, fn), eid ))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]