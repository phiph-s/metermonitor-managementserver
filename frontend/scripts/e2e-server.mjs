import http from 'http';
import net from 'net';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync, readFileSync, rmSync, mkdirSync } from 'fs';
import { spawn } from 'child_process';
import aedes from 'aedes';
import { WebSocketServer } from 'ws';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..', '..');

const MQTT_PORT = Number.parseInt(process.env.E2E_MQTT_PORT || '1889', 10);
const HA_PORT = Number.parseInt(process.env.E2E_HA_PORT || '1888', 10);
const BACKEND_PORT = Number.parseInt(process.env.E2E_BACKEND_PORT || '8070', 10);

const settingsPath = path.join(rootDir, 'settings.e2e.json');
const dbPath = path.join(rootDir, 'data', 'e2e-watermeters.sqlite');
const datasetDir = path.join(rootDir, 'data', 'output_dataset_e2e');
const imagePath = path.join(rootDir, 'test', 'img', 'img.png');

if (existsSync(dbPath)) {
  rmSync(dbPath);
}
if (existsSync(datasetDir)) {
  rmSync(datasetDir, { recursive: true, force: true });
}
mkdirSync(datasetDir, { recursive: true });

const imageBuffer = readFileSync(imagePath);

const mqttBroker = aedes();
const mqttServer = net.createServer(mqttBroker.handle);
mqttServer.listen(MQTT_PORT, '127.0.0.1', () => {
  console.log(`[E2E] MQTT broker listening on 127.0.0.1:${MQTT_PORT}`);
});

const haStates = [
  {
    entity_id: 'camera.test_meter',
    state: 'on',
    attributes: {
      friendly_name: 'Test Camera',
    },
  },
  {
    entity_id: 'light.test_flash',
    state: 'off',
    attributes: {
      friendly_name: 'Test Flash',
    },
  },
];

const haServer = http.createServer((req, res) => {
  const { url, method } = req;

  if (url === '/api/states') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(haStates));
    return;
  }

  if (url === '/api/config') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ location_name: 'Test HA' }));
    return;
  }

  if (url && url.startsWith('/api/camera_proxy/')) {
    res.writeHead(200, { 'Content-Type': 'image/jpeg' });
    res.end(imageBuffer);
    return;
  }

  if (url && url.startsWith('/api/services/light/')) {
    if (method !== 'POST') {
      res.writeHead(405, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Method not allowed' }));
      return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ result: 'ok' }));
    return;
  }

  if (url === '/test-image.png' || url === '/test-image.jpg') {
    res.writeHead(200, { 'Content-Type': 'image/png' });
    res.end(imageBuffer);
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

const wss = new WebSocketServer({ noServer: true });
haServer.on('upgrade', (req, socket, head) => {
  if (req.url === '/api/websocket') {
    wss.handleUpgrade(req, socket, head, (ws) => {
      wss.emit('connection', ws, req);
    });
  } else {
    socket.destroy();
  }
});

wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type: 'auth_required', ha_version: '2024.1.0' }));

  ws.on('message', (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }

    if (msg.type === 'auth') {
      ws.send(JSON.stringify({ type: 'auth_ok' }));
      return;
    }

    if (msg.type === 'config/entity_registry/list') {
      ws.send(
        JSON.stringify({
          id: msg.id || 1,
          type: 'result',
          success: true,
          result: [],
        })
      );
    }
  });
});

haServer.listen(HA_PORT, '127.0.0.1', () => {
  console.log(`[E2E] Fake HA/HTTP server listening on 127.0.0.1:${HA_PORT}`);
});

const backend = spawn('python3', ['run.py'], {
  cwd: rootDir,
  env: {
    ...process.env,
    METERMONITOR_SETTINGS: settingsPath,
    PYTHONUNBUFFERED: '1',
  },
  stdio: 'inherit',
});

console.log(`[E2E] Backend started on 127.0.0.1:${BACKEND_PORT}`);

const shutdown = () => {
  console.log('[E2E] Shutting down...');
  mqttServer.close();
  mqttBroker.close();
  haServer.close();
  wss.close();
  if (backend && !backend.killed) {
    backend.kill('SIGTERM');
  }
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
process.on('exit', shutdown);
