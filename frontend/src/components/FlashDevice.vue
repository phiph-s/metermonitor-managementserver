<template>
  <n-flex class="flash-device" justify="space-around">
    <div style="max-width: 800px;">
      <n-flex style="width: 100%;">
        <n-steps :current="step" size="small" class="flash-steps">
          <n-step title="Select"    description="Choose release & board" />
          <n-step title="Configure" description="Set credentials" />
          <n-step title="Download"  description="Fetch firmware" />
          <n-step title="Flash"     description="Flash via WebSerial" />
        </n-steps>
      </n-flex>

      <!-- ── Step 1: Select ─────────────────────────────────────────────── -->
      <div v-if="step === 1" class="step-body">
        <n-card size="small">
          <template #header>Firmware Release</template>
          <n-space vertical size="medium">
            <n-alert v-if="releasesError" type="error" :title="releasesError" />
            <n-spin v-if="loadingReleases" />
            <template v-else-if="releases.length === 0 && !releasesError">
              <n-empty description="No releases found in the firmware repository." style="padding:32px 0;" />
            </template>
            <template v-else>
              <div class="config-row">
                <span class="config-label-text">Release</span>
                <n-select
                  v-model:value="selectedTag"
                  :options="releases.map(r => ({ label: r.name || r.tag, value: r.tag }))"
                  style="width:220px;"
                  size="small"
                />
              </div>
              <div class="config-row">
                <span class="config-label-text">Board</span>
                <n-select
                  v-model:value="selectedBoard"
                  :options="selectedRelease?.boards.map(b => ({ label: boardLabels[b] || b, value: b })) ?? []"
                  :disabled="!selectedTag"
                  style="width:300px;"
                  size="small"
                />
              </div>
            </template>
          </n-space>
          <template #footer>
            <n-flex justify="end">
              <n-button type="primary" :disabled="!selectedTag || !selectedBoard" @click="step = 2">
                Next: Configure →
              </n-button>
            </n-flex>
          </template>
        </n-card>
      </div>

      <!-- ── Step 2: Configure ──────────────────────────────────────────── -->
      <div v-if="step === 2" class="step-body">
        <n-card size="small">
          <template #header>
            <n-flex align="center" justify="space-between">
              <span>Device Configuration</span>
              <n-flex align="center" gap="8">
                <n-text v-if="saveNotice" depth="2" style="font-size:12px;color:#18a058;">{{ saveNotice }}</n-text>
                <n-button v-if="hasSavedConfig" size="tiny" @click="loadSavedConfig">Load saved</n-button>
                <n-button size="tiny" type="primary" @click="saveConfig">Save for later</n-button>
              </n-flex>
            </n-flex>
          </template>

          <n-collapse :default-expanded-names="['wifi','mqtt','general']" class="config-collapse">
            <!-- WiFi -->
            <n-collapse-item name="wifi">
              <template #header>
                <n-flex align="center" gap="8">
                  <n-icon size="15"><WifiOutlined /></n-icon>
                  <span class="section-title">WiFi</span>
                </n-flex>
              </template>
              <div class="section-options">
                <div class="config-row">
                  <span class="config-label-text">SSID</span>
                  <n-input v-model:value="cfg.wifi_ssid" size="small" style="width:260px;" placeholder="My Network" />
                </div>
                <div class="config-row">
                  <span class="config-label-text">Password</span>
                  <n-input v-model:value="cfg.wifi_pass" type="password" show-password-on="click" size="small" style="width:260px;" />
                </div>
              </div>
            </n-collapse-item>

            <!-- MQTT -->
            <n-collapse-item name="mqtt">
              <template #header>
                <n-flex align="center" gap="8">
                  <n-icon size="15"><CloudOutlined /></n-icon>
                  <span class="section-title">MQTT</span>
                </n-flex>
              </template>
              <div class="section-options">
                <div class="config-row">
                  <span class="config-label-text">Broker URL</span>
                  <n-input v-model:value="cfg.mqtt_url" size="small" style="width:260px;" placeholder="mqtt://192.168.1.1:1883" />
                </div>
                <div class="config-row">
                  <span class="config-label-text">Username</span>
                  <n-input v-model:value="cfg.mqtt_user" size="small" style="width:260px;" />
                </div>
                <div class="config-row">
                  <span class="config-label-text">Password</span>
                  <n-input v-model:value="cfg.mqtt_pass" type="password" show-password-on="click" size="small" style="width:260px;" />
                </div>
                <div class="config-row">
                  <n-flex align="center" gap="6">
                    <span class="config-label-text">Topic</span>
                    <n-tooltip trigger="hover">
                      <template #trigger><n-icon size="13" class="help-icon"><HelpOutlineOutlined /></n-icon></template>
                      The MQTT topic the device publishes images to.<br>
                      Commands are received on {topic}/cmd/{capture,flash,interval}.
                    </n-tooltip>
                  </n-flex>
                  <n-input v-model:value="cfg.mqtt_topic" size="small" style="width:260px;" placeholder="MeterMonitor/meter" />
                </div>
              </div>
            </n-collapse-item>

            <!-- General -->
            <n-collapse-item name="general">
              <template #header>
                <n-flex align="center" gap="8">
                  <n-icon size="15"><SettingsOutlined /></n-icon>
                  <span class="section-title">General</span>
                </n-flex>
              </template>
              <div class="section-options">
                <div class="config-row">
                  <span class="config-label-text">Meter name</span>
                  <n-input v-model:value="cfg.meter_name" size="small" style="width:200px;" placeholder="meter" />
                </div>
                <div class="config-row">
                  <span class="config-label-text">Capture interval (s)</span>
                  <n-input-number v-model:value="cfg.interval" size="small" :min="5" :max="3600" style="width:120px;" />
                </div>
                <div class="config-row">
                  <span class="config-label-text">Flash LED</span>
                  <n-switch v-model:value="cfg.flash_en" />
                </div>
                <div class="config-row">
                  <n-flex align="center" gap="6">
                    <span class="config-label-text">Exposure delay (ms)</span>
                    <n-tooltip trigger="hover">
                      <template #trigger><n-icon size="13" class="help-icon"><HelpOutlineOutlined /></n-icon></template>
                      Time the LED stays on before the photo is taken. 0 = no delay.
                    </n-tooltip>
                  </n-flex>
                  <n-input-number v-model:value="cfg.flash_delay_ms" size="small" :min="0" :max="5000" :step="50" style="width:120px;" :disabled="!cfg.flash_en" />
                </div>
              </div>
            </n-collapse-item>
          </n-collapse>

          <template #footer>
            <n-flex justify="space-between">
              <n-button @click="step = 1">← Back</n-button>
              <n-button type="primary" @click="step = 3">Next: Download →</n-button>
            </n-flex>
          </template>
        </n-card>
      </div>

      <!-- ── Step 3: Download ───────────────────────────────────────────── -->
      <div v-if="step === 3" class="step-body">
        <n-card size="small">
          <template #header>Download Firmware</template>
          <n-space vertical size="medium">
            <div class="status-row">
              <n-icon color="#7e8798"><DeveloperBoardOutlined /></n-icon>
              <span>{{ boardLabels[selectedBoard] || selectedBoard }} — {{ selectedTag }}</span>
            </div>
            <div v-if="downloadDone" class="status-row">
              <n-icon color="#18a058"><CheckCircleOutlined /></n-icon>
              <span>Firmware cached on server.</span>
            </div>
            <div v-else-if="downloading" class="status-row">
              <n-spin :size="16" />
              <span>Downloading from GitHub…</span>
            </div>
            <n-alert v-if="downloadError" type="error" :title="downloadError" />
            <n-button v-if="!downloadDone" type="primary" :loading="downloading" @click="doDownload">
              Download Firmware
            </n-button>
          </n-space>
          <template #footer>
            <n-flex justify="space-between">
              <n-button @click="step = 2">← Back</n-button>
              <n-button type="primary" :disabled="!downloadDone" @click="step = 4">Next: Flash →</n-button>
            </n-flex>
          </template>
        </n-card>
      </div>

      <!-- ── Step 4: Flash ──────────────────────────────────────────────── -->
      <div v-if="step === 4" class="step-body">
        <n-card size="small">
          <template #header>Flash via WebSerial</template>
          <n-space vertical size="medium">
            <n-alert v-if="!webSerialSupported" type="warning" title="WebSerial not supported">
              Please use <strong>Chrome</strong> or <strong>Edge</strong> (v89+).
            </n-alert>
            <template v-else>
              <n-alert v-if="resetMode === 'auto'" type="info" title="Auto reset">
                The device will be reset into bootloader mode automatically via DTR/RTS.
                If connection fails, switch to <strong>Manual</strong> mode.
              </n-alert>
              <n-alert v-else type="warning" title="Manual bootloader mode">
                <ol style="margin:4px 0;padding-left:18px;">
                  <li>Hold the <strong>BOOT</strong> button on the device</li>
                  <li>Click <strong>Connect &amp; Flash</strong> and select the port</li>
                  <li>Release <strong>BOOT</strong> once you see "Connecting…" in the log</li>
                </ol>
              </n-alert>
            </template>
            <n-flex v-if="!flashing && !flashSuccess" align="center" gap="8" wrap>
              <n-button type="primary" :disabled="!webSerialSupported" @click="doFlash">
                Connect &amp; Flash
              </n-button>
              <n-select v-model:value="baudRate" :options="BAUD_RATES" size="small" style="width:130px;" />
              <n-text depth="3" style="font-size:11px;">baud rate</n-text>
              <n-radio-group v-model:value="resetMode" size="small">
                <n-radio-button value="auto">Auto reset</n-radio-button>
                <n-radio-button value="manual">Manual</n-radio-button>
              </n-radio-group>
              <n-button @click="step = 3">← Back</n-button>
            </n-flex>
            <div v-if="flashProgress !== null" class="flash-progress">
              <n-progress type="line" :percentage="flashProgress" :height="10" :border-radius="5" :processing="flashing" />
              <span class="flash-progress-label">{{ flashProgressLabel }}</span>
            </div>
            <div v-if="flashLog.length > 0" class="flash-log">
              <div v-for="(line, i) in flashLog" :key="i" class="flash-log-line">{{ line }}</div>
            </div>
            <n-alert v-if="flashSuccess" type="success" title="Flashed successfully! Device will restart." />
            <n-alert v-if="flashError" type="error" :title="flashError" />
            <n-button v-if="flashSuccess || flashError" @click="resetFlash">Flash again</n-button>

            <n-divider style="margin:4px 0;" />

            <n-flex align="center" gap="8">
              <n-button v-if="!monitorActive" :disabled="!webSerialSupported || flashing" size="small" @click="startMonitor">
                View Device Logs
              </n-button>
              <n-button v-else size="small" type="error" @click="stopMonitor">Stop Monitor</n-button>
              <n-select v-model:value="monitorBaudRate" :options="BAUD_RATES" size="small" style="width:130px;" :disabled="monitorActive" />
              <n-text depth="3" style="font-size:11px;">monitor baud</n-text>
            </n-flex>
            <div v-if="monitorActive || monitorOutput" class="build-terminal" ref="monitorEl">
              <pre>{{ monitorOutput }}</pre>
            </div>
          </n-space>
        </n-card>
      </div>
    </div>
  </n-flex>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue';
import {
  NCard, NButton, NSpace, NFlex, NSteps, NStep,
  NInput, NInputNumber, NSwitch, NEmpty, NAlert, NProgress,
  NIcon, NTooltip, NText, NSpin, NSelect, NDivider,
  NRadioGroup, NRadioButton, NCollapse, NCollapseItem,
} from 'naive-ui';
import {
  CheckCircleOutlined,
  HelpOutlineOutlined,
  WifiOutlined,
  CloudOutlined,
  SettingsOutlined,
  DeveloperBoardOutlined,
} from '@vicons/material';
import { useAuthStore } from '@/stores/authStore';
import { apiService } from '@/services/api.js';

const NVS_OFFSET  = 0x9000;
const SAVED_KEY   = 'mm_nvs_flash_config';
const host        = import.meta.env.VITE_HOST || '';
const authStore   = useAuthStore();

// ── step state ─────────────────────────────────────────────────────────────────
const step = ref(1);

// ── step 1 ─────────────────────────────────────────────────────────────────────
const releases       = ref([]);
const boardLabels    = ref({});
const loadingReleases = ref(false);
const releasesError  = ref('');
const selectedTag    = ref('');
const selectedBoard  = ref('');

const selectedRelease = computed(() =>
  releases.value.find(r => r.tag === selectedTag.value) ?? null
);

async function loadReleases() {
  loadingReleases.value = true;
  releasesError.value   = '';
  try {
    const data = await apiService.getJson('api/flash/releases');
    releases.value    = data.releases    ?? [];
    boardLabels.value = data.board_labels ?? {};
    if (releases.value.length) {
      selectedTag.value   = releases.value[0].tag;
      selectedBoard.value = releases.value[0].boards[0] ?? '';
    }
  } catch (e) {
    releasesError.value = e.message;
  } finally {
    loadingReleases.value = false;
  }
}

// ── step 2 ─────────────────────────────────────────────────────────────────────
const DEFAULT_CFG = () => ({
  wifi_ssid:      '',
  wifi_pass:      '',
  mqtt_url:       'mqtt://192.168.1.1:1883',
  mqtt_user:      '',
  mqtt_pass:      '',
  mqtt_topic:     'MeterMonitor/meter',
  meter_name:     'meter',
  interval:       30,
  flash_en:       true,
  flash_delay_ms: 100,
});

const cfg         = ref(DEFAULT_CFG());
const saveNotice  = ref('');
const hasSavedConfig = ref(!!localStorage.getItem(SAVED_KEY));

function saveConfig() {
  localStorage.setItem(SAVED_KEY, JSON.stringify(cfg.value));
  hasSavedConfig.value = true;
  saveNotice.value = 'Saved!';
  setTimeout(() => { saveNotice.value = ''; }, 1800);
}

function loadSavedConfig() {
  try {
    const saved = JSON.parse(localStorage.getItem(SAVED_KEY) ?? '{}');
    cfg.value = { ...DEFAULT_CFG(), ...saved };
    saveNotice.value = 'Loaded!';
    setTimeout(() => { saveNotice.value = ''; }, 1800);
  } catch { /* ignore */ }
}

// ── step 3 ─────────────────────────────────────────────────────────────────────
const downloading  = ref(false);
const downloadDone = ref(false);
const downloadError = ref('');

async function doDownload() {
  downloading.value  = true;
  downloadError.value = '';
  try {
    await apiService.postJson('api/flash/download', {
      tag:   selectedTag.value,
      board: selectedBoard.value,
    });
    downloadDone.value = true;
  } catch (e) {
    downloadError.value = e.message;
  } finally {
    downloading.value = false;
  }
}

// ── step 4: flash ──────────────────────────────────────────────────────────────
const BAUD_RATES = [
  { label: '921600', value: 921600 },
  { label: '460800', value: 460800 },
  { label: '230400', value: 230400 },
  { label: '115200', value: 115200 },
];

const webSerialSupported  = ref('serial' in navigator);
const baudRate            = ref(460800);
const resetMode           = ref('auto');
const flashing            = ref(false);
const flashSuccess        = ref(false);
const flashError          = ref('');
const flashLog            = ref([]);
const flashProgress       = ref(null);
const flashProgressLabel  = ref('');

const monitorActive   = ref(false);
const monitorOutput   = ref('');
const monitorBaudRate = ref(115200);
const monitorEl       = ref(null);
let _monitorReader = null;
let _monitorPort   = null;

function flashLog_push(line) { flashLog.value.push(line); }

async function doFlash() {
  flashing.value    = true;
  flashError.value  = '';
  flashLog.value    = [];
  flashProgress.value = 0;
  flashSuccess.value  = false;

  let transport = null;

  try {
    // 1. Request serial port immediately — must be the first await to stay
    //    within the browser's user-gesture token window.
    flashLog_push('Initializing WebSerial…');
    const port = await navigator.serial.requestPort();

    // 2. Get firmware flash args
    flashLog_push('Fetching flash configuration…');
    const flashArgs = await apiService.getJson(
      `api/flash/flash-args?tag=${encodeURIComponent(selectedTag.value)}&board=${encodeURIComponent(selectedBoard.value)}`
    );
    if (!flashArgs.binaries?.length) throw new Error('No binaries found — try re-downloading.');

    // 3. Generate NVS binary from config
    flashLog_push('Generating NVS partition…');
    const nvsResp = await fetch(`${host}api/flash/nvs`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', secret: authStore.secret },
      body:    JSON.stringify(cfg.value),
    });
    if (!nvsResp.ok) throw new Error(`NVS generation failed: ${nvsResp.status}`);
    const nvsBuf = await nvsResp.arrayBuffer();

    // 4. Download firmware binaries
    flashLog_push(`Downloading ${flashArgs.binaries.length} firmware file(s)…`);
    const fileArray = [];
    for (const bin of flashArgs.binaries) {
      flashLog_push(`  ↓ ${bin.filename} @ 0x${bin.offset.toString(16)} (${(bin.size / 1024).toFixed(1)} KB)`);
      const resp = await fetch(`${host}api/flash/binaries/${bin.path}`, {
        headers: { secret: authStore.secret },
      });
      if (!resp.ok) throw new Error(`Failed to download ${bin.filename}: ${resp.status}`);
      fileArray.push({ data: new Uint8Array(await resp.arrayBuffer()), address: bin.offset });
    }

    // 5. Add NVS partition
    flashLog_push(`  + nvs.bin @ 0x${NVS_OFFSET.toString(16)} (${(nvsBuf.byteLength / 1024).toFixed(1)} KB)`);
    fileArray.push({ data: new Uint8Array(nvsBuf), address: NVS_OFFSET });

    // 6. Connect and flash
    const { ESPLoader, Transport } = await import('esptool-js');
    transport  = new Transport(port, false);
    const terminal = {
      clean:     () => {},
      writeLine: (d) => flashLog_push(d),
      write:     (d) => {
        if (flashLog.value.length) flashLog.value[flashLog.value.length - 1] += d;
        else flashLog_push(d);
      },
    };

    const isManual = resetMode.value === 'manual';
    flashLog_push(isManual ? 'Connecting (manual mode)…' : 'Connecting to device…');
    const loader = new ESPLoader({ transport, baudrate: baudRate.value, terminal, debugLogging: false });
    const chip   = await loader.main(isManual ? 'no_reset' : 'default_reset');
    flashLog_push(`Connected: ${chip}`);

    const total = fileArray.length;
    flashLog_push('Writing flash…');
    await loader.writeFlash({
      fileArray,
      flashMode: flashArgs.flash_mode ?? 'keep',
      flashFreq: flashArgs.flash_freq ?? 'keep',
      flashSize: flashArgs.flash_size ?? 'keep',
      eraseAll:  false,
      compress:  true,
      reportProgress: (fileIndex, written, size) => {
        const pct = size > 0 ? written / size : 1;
        flashProgress.value      = Math.round(((fileIndex + pct) / total) * 100);
        flashProgressLabel.value = `File ${fileIndex + 1}/${total}: ${Math.round(pct * 100)}%`;
      },
      calculateMD5Hash: () => '',
    });

    flashLog_push('Resetting device…');
    await loader.after('hard_reset');
    flashProgress.value      = 100;
    flashProgressLabel.value = 'Done!';
    flashSuccess.value = true;
  } catch (e) {
    if (e.name !== 'NotFoundError') {
      flashError.value = e.message || String(e);
      flashLog_push(`Error: ${flashError.value}`);
    }
  } finally {
    if (transport) { try { await transport.disconnect(); } catch { /* ignore */ } }
    flashing.value = false;
  }
}

function resetFlash() {
  flashSuccess.value  = false;
  flashError.value    = '';
  flashLog.value      = [];
  flashProgress.value = null;
}

// ── serial monitor ─────────────────────────────────────────────────────────────
async function startMonitor() {
  monitorActive.value = true;
  monitorOutput.value = '';
  try {
    const port = await navigator.serial.requestPort();
    _monitorPort = port;
    await port.open({ baudRate: monitorBaudRate.value });
    const reader  = port.readable.getReader();
    _monitorReader = reader;
    const decoder = new TextDecoder();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        monitorOutput.value += decoder.decode(value, { stream: true });
        await nextTick();
        if (monitorEl.value) monitorEl.value.scrollTop = monitorEl.value.scrollHeight;
      }
    } catch (e) {
      if (monitorActive.value) monitorOutput.value += `\n[Error: ${e.message}]\n`;
    } finally {
      reader.releaseLock();
    }
  } catch (e) {
    if (e.name !== 'NotFoundError') monitorOutput.value += `Error: ${e.message}\n`;
  } finally {
    monitorActive.value = false;
    _monitorReader = null;
    if (_monitorPort) { try { await _monitorPort.close(); } catch { /* ignore */ } _monitorPort = null; }
  }
}

async function stopMonitor() {
  monitorActive.value = false;
  if (_monitorReader) { try { await _monitorReader.cancel(); } catch { /* ignore */ } _monitorReader = null; }
  if (_monitorPort)   { try { await _monitorPort.close();    } catch { /* ignore */ } _monitorPort   = null; }
}

// ── init ───────────────────────────────────────────────────────────────────────
onMounted(() => {
  loadReleases();
  if (localStorage.getItem(SAVED_KEY)) loadSavedConfig();
});
</script>

<style scoped>
.flash-device { width: 100%; }
.flash-steps  { margin-bottom: 20px; }
.step-body    { margin-top: 4px; width: 800px; }

.build-terminal {
  background: #111827;
  border-radius: 8px;
  padding: 12px 16px;
  height: 420px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.5;
}
.build-terminal pre { margin: 0; white-space: pre-wrap; word-break: break-all; color: #e5e7eb; }

.config-collapse { margin: -4px 0; }
.section-title   { font-size: 13px; font-weight: 600; }

.section-options {
  display: flex;
  flex-direction: column;
  padding: 4px 0 8px 0;
}

.config-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 2px;
  border-bottom: 1px solid rgba(255,255,255,.05);
  gap: 12px;
  min-height: 38px;
}
.config-row:last-child { border-bottom: none; }

.config-label-text { font-size: 13px; font-weight: 500; }
.help-icon         { opacity: .45; cursor: help; }

.status-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }

.flash-progress       { display: flex; flex-direction: column; gap: 4px; }
.flash-progress-label { font-size: 12px; opacity: .6; }

.flash-log {
  background: rgba(0,0,0,.3);
  border-radius: 8px;
  padding: 10px 14px;
  max-height: 200px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.6;
}
.flash-log-line { white-space: pre-wrap; word-break: break-all; }

.light-mode .flash-log      { background: rgba(0,0,0,.06); color: #111; }
.light-mode .build-terminal { background: #1f2937; }
.light-mode .config-row     { border-bottom-color: rgba(0,0,0,.07); }
</style>
