/**
 * E2E test server for Playwright.
 * Starts three services required by the test suite:
 *   1. Aedes MQTT broker     → mqtt://127.0.0.1:1889
 *   2. Mock HA HTTP server   → http://127.0.0.1:1888
 *      - GET /test-image.png             (image for HTTP sources)
 *      - GET /api/states                 (camera entity list)
 *      - GET /api/camera_proxy/<id>      (camera snapshot)
 *   3. Python backend        → http://127.0.0.1:8070  (settings.e2e.json)
 */

import { createServer as createTcpServer } from 'net';
import { createServer as createHttpServer } from 'http';
import { readFileSync, existsSync, rmSync } from 'fs';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import Aedes from 'aedes';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');

// ── Test image ────────────────────────────────────────────────────────────────
const imagePath = path.join(repoRoot, 'test', 'img', 'img.png');
const imageBytes = readFileSync(imagePath);

// ── 1. MQTT broker (aedes) ────────────────────────────────────────────────────
const aedes = new Aedes();
const mqttServer = createTcpServer(aedes.handle);
mqttServer.listen(1889, '127.0.0.1', () => {
  console.log('[e2e] MQTT broker   → mqtt://127.0.0.1:1889');
});

// ── 2. Mock HA HTTP server ────────────────────────────────────────────────────
const haServer = createHttpServer((req, res) => {
  // HTTP source image
  if (req.url === '/test-image.png') {
    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(imageBytes);
    return;
  }

  // HA states — returns one camera entity so the dropdown shows "Test Camera"
  if (req.url === '/api/states') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify([
      {
        entity_id: 'camera.test_camera',
        state: 'idle',
        attributes: { friendly_name: 'Test Camera' },
      },
    ]));
    return;
  }

  // HA camera snapshot
  if (req.url.startsWith('/api/camera_proxy/')) {
    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(imageBytes);
    return;
  }

  // HA entity registry (used by flash-entity suggestion — just return empty)
  if (req.url.startsWith('/api/')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify([]));
    return;
  }

  res.writeHead(404);
  res.end('Not found');
});

haServer.listen(1888, '127.0.0.1', () => {
  console.log('[e2e] Mock HA server → http://127.0.0.1:1888');
});

// ── 3. Python backend ─────────────────────────────────────────────────────────
const settingsPath = path.join(repoRoot, 'settings.e2e.json');

// Remove stale e2e database so each run starts clean
const dbPath = path.join(repoRoot, 'data', 'e2e-watermeters.sqlite');
if (existsSync(dbPath)) {
  rmSync(dbPath);
  console.log('[e2e] Removed stale e2e database');
}

const python = spawn('python3', ['run.py'], {
  cwd: repoRoot,
  stdio: 'inherit',
  env: { ...process.env, METERMONITOR_SETTINGS: settingsPath },
});

python.on('error', (err) => {
  console.error('[e2e] Failed to start Python backend:', err.message);
  process.exit(1);
});

python.on('exit', (code) => {
  console.log(`[e2e] Python backend exited (code ${code})`);
  process.exit(code ?? 1);
});

// ── Cleanup ───────────────────────────────────────────────────────────────────
const shutdown = () => {
  python.kill('SIGTERM');
  mqttServer.close();
  haServer.close();
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
