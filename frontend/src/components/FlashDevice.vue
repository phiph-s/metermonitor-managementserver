<template>
  <div class="flash-device">
    <!-- Stepper -->
    <n-steps :current="step" size="small" class="flash-steps">
      <n-step title="Prepare" description="Clone / update firmware" />
      <n-step title="Configure" description="Set device options" />
      <n-step title="Build" description="Compile firmware" />
      <n-step title="Flash" description="Flash via WebSerial" />
    </n-steps>

    <!-- ───────────────────────────── Step 1: Prepare ───────────────────────────── -->
    <div v-if="step === 1" class="step-body">
      <n-card size="small">
        <template #header>Firmware Repository</template>
        <n-space vertical size="medium">
          <div v-if="firmwareReady" class="status-row">
            <n-icon color="#18a058"><CheckCircleOutlined /></n-icon>
            <span>Repository available at <code>{{ FIRMWARE_DIR }}</code></span>
          </div>
          <div v-else class="status-row">
            <n-icon color="#f0a020"><InfoOutlined /></n-icon>
            <span>Repository not yet cloned.</span>
          </div>
          <n-space>
            <n-button type="primary" :loading="preparing" @click="doPrepare">
              {{ firmwareReady ? 'Pull latest changes' : 'Clone repository' }}
            </n-button>
            <n-button v-if="firmwareReady" :disabled="preparing" @click="step = 2">
              Next: Configure →
            </n-button>
          </n-space>
          <div v-if="prepareOutput" class="log-block"><pre>{{ prepareOutput }}</pre></div>
          <n-alert v-if="prepareError" type="error" :title="prepareError" />
        </n-space>
      </n-card>
    </div>

    <!-- ───────────────────────────── Step 2: Configure ─────────────────────────── -->
    <div v-if="step === 2" class="step-body">
      <n-spin :show="loadingKconfig">
        <n-card size="small">
          <template #header>
            <n-flex align="center" justify="space-between">
              <span>MeterMonitor Configuration</span>
              <n-flex align="center" gap="8">
                <n-text v-if="saveNotice" depth="2" style="font-size:12px;color:#18a058;">
                  {{ saveNotice }}
                </n-text>
                <n-button
                  v-if="hasSavedConfig"
                  size="tiny"
                  @click="loadSavedConfig"
                >
                  Load saved
                </n-button>
                <n-button
                  size="tiny"
                  type="primary"
                  :disabled="kconfigOptions.length === 0"
                  @click="saveConfig"
                >
                  Save for later
                </n-button>
                <n-divider vertical style="height:16px;margin:0 2px;" />
                <n-flex align="center" gap="4">
                  <n-switch v-model:value="advancedMode" size="small" />
                  <n-text depth="3" style="font-size:12px;">Advanced</n-text>
                </n-flex>
              </n-flex>
            </n-flex>
          </template>

          <n-empty
            v-if="!loadingKconfig && kconfigOptions.length === 0"
            description="No MeterMonitor config options found in the repository."
            style="padding: 32px 0;"
          />

          <!-- Sections -->
          <n-collapse
            v-else
            :default-expanded-names="SECTION_ORDER"
            class="config-collapse"
          >
            <n-collapse-item
              v-for="sec in optionsBySection"
              :key="sec.key"
              :name="sec.key"
            >
              <template #header>
                <n-flex align="center" gap="8">
                  <n-icon size="15"><component :is="SECTION_ICONS[sec.key]" /></n-icon>
                  <span class="section-title">{{ sec.label }}</span>
                  <n-text depth="3" style="font-size:11px;">{{ sec.options.length }}</n-text>
                </n-flex>
              </template>

              <div class="section-options">
                <div
                  v-for="opt in sec.options"
                  :key="opt.name"
                  class="config-row"
                >
                  <!-- Label column -->
                  <div class="config-label-col">
                    <span class="config-label-text">{{ opt.label }}</span>
                    <n-tooltip
                      v-if="opt.help"
                      trigger="hover"
                      placement="right"
                      :style="{ maxWidth: '300px' }"
                    >
                      <template #trigger>
                        <n-icon size="13" class="help-icon"><HelpOutlineOutlined /></n-icon>
                      </template>
                      <span style="white-space:pre-wrap; font-size:12px;">{{ opt.help }}</span>
                    </n-tooltip>
                    <span class="config-name-badge">{{ opt.name }}</span>
                  </div>

                  <!-- Input column -->
                  <div class="config-input-col">
                    <!-- choice → dropdown -->
                    <n-select
                      v-if="opt.type === 'choice'"
                      v-model:value="configValues[opt.name]"
                      size="small"
                      :options="opt.choices.map(c => ({ label: c.label, value: c.name }))"
                      style="min-width: 220px;"
                    />

                    <!-- bool → switch -->
                    <n-switch v-else-if="opt.type === 'bool'" v-model:value="configValues[opt.name]" />

                    <!-- password → masked input -->
                    <n-input
                      v-else-if="isPassword(opt)"
                      v-model:value="configValues[opt.name]"
                      type="password"
                      show-password-on="click"
                      size="small"
                      :placeholder="opt.default !== null && opt.default !== undefined ? String(opt.default) : ''"
                      style="width: 260px;"
                    />

                    <!-- int / hex → number -->
                    <n-input-number
                      v-else-if="opt.type === 'int' || opt.type === 'hex'"
                      v-model:value="configValues[opt.name]"
                      size="small"
                      :min="opt.range ? Number(opt.range[0]) : undefined"
                      :max="opt.range ? Number(opt.range[1]) : undefined"
                      :placeholder="opt.default !== null ? String(opt.default) : ''"
                      style="width: 130px;"
                    />

                    <!-- string → text -->
                    <n-input
                      v-else
                      v-model:value="configValues[opt.name]"
                      size="small"
                      :placeholder="opt.default !== null && opt.default !== undefined ? String(opt.default) : ''"
                      style="width: 260px;"
                    />
                  </div>
                </div>
              </div>
            </n-collapse-item>
          </n-collapse>

          <template #footer>
            <n-flex justify="space-between">
              <n-button @click="step = 1">← Back</n-button>
              <n-button type="primary" :disabled="kconfigOptions.length === 0" @click="doBuild">
                Build Firmware →
              </n-button>
            </n-flex>
          </template>
        </n-card>
      </n-spin>
    </div>

    <!-- ───────────────────────────── Step 3: Build ──────────────────────────────── -->
    <div v-if="step === 3" class="step-body">
      <n-card size="small">
        <template #header>
          <n-flex align="center" gap="8">
            <span>Build Output</span>
            <n-spin v-if="building" :size="14" />
            <n-tag v-if="buildDone && buildSuccess" type="success" size="small">Success</n-tag>
            <n-tag v-if="buildDone && !buildSuccess" type="error" size="small">Failed</n-tag>
          </n-flex>
        </template>
        <div class="build-terminal" ref="buildTerminalEl">
          <pre>{{ buildOutput }}</pre>
        </div>
        <template #footer>
          <n-flex justify="space-between">
            <n-button @click="step = 2">← Back to Configure</n-button>
            <n-space v-if="buildDone">
              <n-button v-if="!buildSuccess" :loading="building" @click="doBuild">Retry Build</n-button>
              <n-button v-if="buildSuccess" type="primary" @click="step = 4">Flash Device →</n-button>
            </n-space>
          </n-flex>
        </template>
      </n-card>
    </div>

    <!-- ───────────────────────────── Step 4: Flash ──────────────────────────────── -->
    <div v-if="step === 4" class="step-body">
      <n-card size="small">
        <template #header>Flash via WebSerial</template>
        <n-space vertical size="medium">
          <n-alert v-if="!webSerialSupported" type="warning" title="WebSerial not supported">
            Please use <strong>Chrome</strong> or <strong>Edge</strong> (v89+).
          </n-alert>
          <n-alert v-else type="info" title="Before connecting">
            <ul style="margin:4px 0;padding-left:18px;">
              <li>Hold the <strong>BOOT</strong> button while plugging in the USB cable</li>
              <li>Release once connected, then select the correct port</li>
            </ul>
          </n-alert>
          <n-space v-if="!flashing && !flashSuccess">
            <n-button type="primary" :disabled="!webSerialSupported" @click="doFlash">
              Connect &amp; Flash
            </n-button>
            <n-button @click="step = 3">← Back to Build</n-button>
          </n-space>
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
        </n-space>
      </n-card>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, onMounted, shallowRef } from 'vue';
import {
  NCard, NButton, NSpace, NFlex, NSteps, NStep,
  NInput, NInputNumber, NSwitch, NEmpty, NAlert, NProgress,
  NIcon, NTooltip, NText, NSpin, NTag, NCollapse, NCollapseItem,
  NSelect, NDivider,
} from 'naive-ui';
import {
  CheckCircleOutlined,
  InfoOutlined,
  HelpOutlineOutlined,
  WifiOutlined,
  CloudOutlined,
  AccessTimeOutlined,
  PhotoCameraOutlined,
  BuildOutlined,
  SettingsOutlined,
  DeveloperBoardOutlined,
} from '@vicons/material';
import { useAuthStore } from '@/stores/authStore';
import { apiService } from '@/services/api.js';

// ── Constants ──────────────────────────────────────────────────────────────────

const FIRMWARE_DIR = '/data/metermonitor-esp';
const host = import.meta.env.VITE_HOST || '';

const SECTION_DEFS = [
  { key: 'Board',    label: 'Board',          pattern: /BOARD/i },
  { key: 'General',  label: 'General',         pattern: /INTERVAL|_NAME$|UNIQUE|IDENTIFYING|DEVICE_NAME|METER_NAME/i },
  { key: 'WiFi',     label: 'WiFi',            pattern: /WIFI|SSID/i },
  { key: 'MQTT',     label: 'MQTT',            pattern: /MQTT|BROKER/i },
  { key: 'Time',     label: 'Time / SNTP',     pattern: /SNTP|NTP|TIME/i },
  { key: 'Camera',   label: 'Camera',          pattern: /CAMERA|FRAME|RESOL/i },
  { key: 'Hardware', label: 'Hardware / GPIO', pattern: /GPIO|PIN|FLASH|LED_STRIP|DONE/i },
  { key: 'Other',    label: 'Other',           pattern: /.*/ },  // catch-all
];

const SECTION_ORDER = SECTION_DEFS.map(d => d.key);

const SECTION_ICONS = {
  Board:    DeveloperBoardOutlined,
  General:  SettingsOutlined,
  WiFi:     WifiOutlined,
  MQTT:     CloudOutlined,
  Time:     AccessTimeOutlined,
  Camera:   PhotoCameraOutlined,
  Hardware: BuildOutlined,
  Other:    SettingsOutlined,
};

const ADVANCED_ONLY = new Set([
  'METER_MONITOR_WIFI_MAXIMUM_RETRY',
  'METER_MONITOR_DONE',
  'METER_MONITOR_DONE_GPIO',
  'METER_MONITOR_FLASH_GPIO',
  'METER_MONITOR_LED_STRIP',
  'METER_MONITOR_SNTP_TIME_SERVER',
  'METER_MONITOR_SNTP_TIME_SYNC_ALWAYS',
]);

// ── Auth ───────────────────────────────────────────────────────────────────────

const authStore = useAuthStore();

// ── Step state ─────────────────────────────────────────────────────────────────

const step = ref(1);

// ── Step 1 ─────────────────────────────────────────────────────────────────────

const firmwareReady = ref(false);
const preparing = ref(false);
const prepareOutput = ref('');
const prepareError = ref('');

// ── Step 2 ─────────────────────────────────────────────────────────────────────

const SAVED_CONFIG_KEY = 'mm_flash_device_config';

const advancedMode = ref(false);
const loadingKconfig = ref(false);
const kconfigOptions = ref([]);   // raw options from backend
const configValues = ref({});     // form values (choice stored as selected name)
const hasSavedConfig = ref(!!localStorage.getItem(SAVED_CONFIG_KEY));
const saveNotice = ref('');

function saveConfig() {
  localStorage.setItem(SAVED_CONFIG_KEY, JSON.stringify(configValues.value));
  hasSavedConfig.value = true;
  saveNotice.value = 'Saved!';
  setTimeout(() => { saveNotice.value = ''; }, 1800);
}

function loadSavedConfig() {
  const raw = localStorage.getItem(SAVED_CONFIG_KEY);
  if (!raw) return;
  try {
    const saved = JSON.parse(raw);
    // Merge: only overwrite keys that exist in current form
    const next = { ...configValues.value };
    for (const [k, v] of Object.entries(saved)) {
      if (k in next) next[k] = v;
    }
    configValues.value = next;
    saveNotice.value = 'Loaded!';
    setTimeout(() => { saveNotice.value = ''; }, 1800);
  } catch { /* ignore malformed storage */ }
}

// ── Step 3 ─────────────────────────────────────────────────────────────────────

const building = ref(false);
const buildOutput = ref('');
const buildDone = ref(false);
const buildSuccess = ref(false);
const buildTerminalEl = ref(null);

// ── Step 4 ─────────────────────────────────────────────────────────────────────

const webSerialSupported = ref('serial' in navigator);
const flashing = ref(false);
const flashSuccess = ref(false);
const flashError = ref('');
const flashLog = ref([]);
const flashProgress = ref(null);
const flashProgressLabel = ref('');

// ── Section logic ──────────────────────────────────────────────────────────────

function getSection(optName) {
  for (const def of SECTION_DEFS) {
    if (def.pattern.test(optName)) return def.key;
  }
  return 'Other';
}

// ── Depends-on evaluation ──────────────────────────────────────────────────────
// Choices are stored as configValues[CHOICE_NAME] = "SELECTED_OPT_NAME"
// For depends_on we need to resolve individual option names (CHOICE_OPT_A → bool)
// resolvedFlat expands choices so evaluateDepends works correctly.

const resolvedFlat = computed(() => {
  const flat = { ...configValues.value };
  for (const opt of kconfigOptions.value) {
    if (opt.type === 'choice') {
      const selected = configValues.value[opt.name];
      for (const c of opt.choices) {
        flat[c.name] = (c.name === selected);
      }
    }
  }
  return flat;
});

function tokenizeDepends(expr) {
  const tokens = [];
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (' \t'.includes(c)) { i++; continue; }
    if (c === '(') { tokens.push({ t: 'LP' }); i++; }
    else if (c === ')') { tokens.push({ t: 'RP' }); i++; }
    else if (c === '!') { tokens.push({ t: 'NOT' }); i++; }
    else if (expr.startsWith('&&', i)) { tokens.push({ t: 'AND' }); i += 2; }
    else if (expr.startsWith('||', i)) { tokens.push({ t: 'OR' }); i += 2; }
    else if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < expr.length && /[A-Za-z0-9_]/.test(expr[j])) j++;
      tokens.push({ t: 'ID', v: expr.slice(i, j) });
      i = j;
    } else { i++; }
  }
  return tokens;
}

function evaluateDepends(expr, flat) {
  if (!expr) return true;
  const tokens = tokenizeDepends(expr);
  let pos = 0;
  const peek = () => tokens[pos];
  const consume = () => tokens[pos++];
  const resolve = (name) => {
    if (!(name in flat)) return true; // unknown → show
    const v = flat[name];
    if (v === true || v === 'y') return true;
    if (v === false || v === 'n' || v === null || v === undefined) return false;
    return typeof v === 'string' ? v !== '' : !!v;
  };
  function prim() {
    const tok = peek();
    if (!tok) return true;
    if (tok.t === 'NOT') { consume(); return !prim(); }
    if (tok.t === 'LP') { consume(); const r = parseOr(); if (peek()?.t === 'RP') consume(); return r; }
    if (tok.t === 'ID') { consume(); return resolve(tok.v); }
    return true;
  }
  function parseAnd() { let v = prim(); while (peek()?.t === 'AND') { consume(); v = prim() && v; } return v; }
  function parseOr()  { let v = parseAnd(); while (peek()?.t === 'OR')  { consume(); v = parseAnd() || v; } return v; }
  return parseOr();
}

// ── Visible options grouped by section ─────────────────────────────────────────

const optionsBySection = computed(() => {
  const flat = resolvedFlat.value;
  const advanced = advancedMode.value;
  const buckets = {};
  for (const opt of kconfigOptions.value) {
    if (!evaluateDepends(opt.depends_on, flat)) continue;
    if (!advanced && ADVANCED_ONLY.has(opt.name)) continue;
    const key = getSection(opt.name);
    if (!buckets[key]) buckets[key] = [];
    buckets[key].push(opt);
  }
  return SECTION_ORDER
    .filter(k => buckets[k]?.length > 0)
    .map(k => ({ key: k, label: SECTION_DEFS.find(d => d.key === k).label, options: buckets[k] }));
});

// ── Helpers ────────────────────────────────────────────────────────────────────

function isPassword(opt) {
  return opt.type === 'string' && /PASS(WORD|WD)?|SECRET/i.test(opt.name);
}

function toNumber(v) { const n = Number(v); return isNaN(n) ? 0 : n; }

function defaultForOpt(opt) {
  if (opt.type === 'choice') {
    return opt.default ?? (opt.choices[0]?.name ?? null);
  }
  if (opt.default !== null && opt.default !== undefined) {
    if (opt.type === 'int' || opt.type === 'hex') return toNumber(opt.default);
    return opt.default;
  }
  if (opt.type === 'bool') return false;
  if (opt.type === 'int' || opt.type === 'hex') return 0;
  return '';
}

function initConfigValues(options, savedValues) {
  const vals = {};
  for (const opt of options) {
    if (opt.type === 'choice') {
      const active = opt.choices.find(c => savedValues[c.name] === true);
      vals[opt.name] = active ? active.name : defaultForOpt(opt);
    } else if (opt.name in savedValues) {
      const raw = savedValues[opt.name];
      if (opt.type === 'int' || opt.type === 'hex') {
        vals[opt.name] = toNumber(raw);
      } else {
        vals[opt.name] = raw;
      }
    } else {
      vals[opt.name] = defaultForOpt(opt);
    }
  }
  return vals;
}

// ── Payload for backend (expand choices → individual booleans) ─────────────────

function buildPayload() {
  const payload = {};
  for (const opt of kconfigOptions.value) {
    if (opt.type === 'choice') {
      const selected = configValues.value[opt.name];
      for (const c of opt.choices) {
        payload[c.name] = (c.name === selected);
      }
    } else {
      payload[opt.name] = configValues.value[opt.name];
    }
  }
  return payload;
}

// ── Step 1: Prepare ────────────────────────────────────────────────────────────

async function doPrepare() {
  preparing.value = true;
  prepareOutput.value = '';
  prepareError.value = '';
  try {
    const result = await apiService.postJson('api/flash/prepare', {});
    prepareOutput.value = result.output || '';
    if (result.success) {
      firmwareReady.value = true;
      await loadKconfig();
    } else {
      prepareError.value = `${result.action} failed – see output above.`;
    }
  } catch (e) {
    prepareError.value = e.message;
  } finally {
    preparing.value = false;
  }
}

// ── Step 2: Load Kconfig ───────────────────────────────────────────────────────

async function loadKconfig() {
  loadingKconfig.value = true;
  try {
    const data = await apiService.getJson('api/flash/kconfig');
    kconfigOptions.value = data.options || [];
    configValues.value = initConfigValues(kconfigOptions.value, data.current_values || {});
  } catch {
    // firmware not ready yet
  } finally {
    loadingKconfig.value = false;
  }
}

// ── Step 3: Build ──────────────────────────────────────────────────────────────

async function doBuild() {
  step.value = 3;
  building.value = true;
  buildOutput.value = '';
  buildDone.value = false;
  buildSuccess.value = false;

  try {
    const response = await fetch(`${host}api/flash/build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', secret: authStore.secret },
      body: JSON.stringify({ config: buildPayload() }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buildOutput.value += decoder.decode(value, { stream: true });
      await nextTick();
      if (buildTerminalEl.value) buildTerminalEl.value.scrollTop = buildTerminalEl.value.scrollHeight;
    }
  } catch (e) {
    buildOutput.value += `\nError: ${e.message}\n`;
  } finally {
    building.value = false;
    buildDone.value = true;
    buildSuccess.value = buildOutput.value.includes('__BUILD_SUCCESS__');
    buildOutput.value = buildOutput.value
      .replace('__BUILD_SUCCESS__\n', '')
      .replace(/__BUILD_FAILED__\d+__\n?/, '');
  }
}

// ── Step 4: Flash ──────────────────────────────────────────────────────────────

function flashLog_push(line) { flashLog.value.push(line); }

async function doFlash() {
  flashing.value = true;
  flashError.value = '';
  flashLog.value = [];
  flashProgress.value = 0;
  flashSuccess.value = false;

  try {
    flashLog_push('Fetching flash configuration...');
    const flashArgs = await apiService.getJson('api/flash/flash-args');
    if (!flashArgs.binaries?.length) throw new Error('No binaries found. Was the build successful?');

    flashLog_push(`Downloading ${flashArgs.binaries.length} binary file(s)...`);
    const fileArray = [];
    for (const bin of flashArgs.binaries) {
      flashLog_push(`  ↓ ${bin.filename} @ 0x${bin.offset.toString(16)} (${(bin.size / 1024).toFixed(1)} KB)`);
      const resp = await fetch(`${host}api/flash/binaries/${bin.path}`, { headers: { secret: authStore.secret } });
      if (!resp.ok) throw new Error(`Failed to download ${bin.filename}: ${resp.status}`);
      const buf = await resp.arrayBuffer();
      fileArray.push({ data: new TextDecoder('latin1').decode(buf), address: bin.offset });
    }

    flashLog_push('Initializing WebSerial...');
    const { ESPLoader, Transport } = await import('esptool-js');
    const port = await navigator.serial.requestPort();
    const transport = new Transport(port, true);
    const terminal = {
      clean: () => {},
      writeLine: (d) => flashLog_push(d),
      write: (d) => {
        if (flashLog.value.length) flashLog.value[flashLog.value.length - 1] += d;
        else flashLog_push(d);
      },
    };

    flashLog_push('Connecting to device...');
    const loader = new ESPLoader({ transport, baudrate: 921600, terminal });
    const chip = await loader.main();
    flashLog_push(`Connected: ${chip}`);

    const total = fileArray.length;
    flashLog_push('Writing flash...');
    await loader.write_flash({
      fileArray,
      flashSize: 'keep',
      eraseAll: false,
      compress: true,
      reportProgress: (fileIndex, written, size) => {
        const pct = size > 0 ? written / size : 1;
        flashProgress.value = Math.round(((fileIndex + pct) / total) * 100);
        flashProgressLabel.value = `File ${fileIndex + 1}/${total}: ${Math.round(pct * 100)}%`;
      },
      calculateMD5Hash: () => '',
    });

    flashLog_push('Resetting device...');
    await transport.disconnect();
    flashProgress.value = 100;
    flashProgressLabel.value = 'Done!';
    flashSuccess.value = true;
  } catch (e) {
    if (e.name !== 'NotFoundError') {
      flashError.value = e.message || String(e);
      flashLog_push(`Error: ${flashError.value}`);
    }
  } finally {
    flashing.value = false;
  }
}

function resetFlash() {
  flashSuccess.value = false;
  flashError.value = '';
  flashLog.value = [];
  flashProgress.value = null;
}

// ── Mount: try to load existing config ─────────────────────────────────────────

onMounted(async () => {
  try {
    await loadKconfig();
    if (kconfigOptions.value.length > 0) firmwareReady.value = true;
  } catch { /* repo not ready */ }
});
</script>

<style scoped>
.flash-device { max-width: 900px; }
.flash-steps  { margin-bottom: 20px; }
.step-body    { margin-top: 4px; }

/* ── log / terminal ── */
.log-block {
  background: rgba(0,0,0,.35);
  border-radius: 8px;
  padding: 10px 14px;
  max-height: 200px;
  overflow-y: auto;
  font-family: monospace;
  font-size: 12px;
  line-height: 1.5;
}
.log-block pre { margin: 0; white-space: pre-wrap; word-break: break-all; }

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

/* ── config form ── */
.config-collapse { margin: -4px 0; }

.section-title { font-size: 13px; font-weight: 600; }

.section-options {
  display: flex;
  flex-direction: column;
  gap: 0;
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

.config-label-col {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  min-width: 0;
  flex-wrap: wrap;
}
.config-label-text  { font-size: 13px; font-weight: 500; }
.config-name-badge  {
  font-size: 10px;
  font-family: monospace;
  opacity: .4;
  background: rgba(255,255,255,.06);
  border-radius: 4px;
  padding: 1px 5px;
  white-space: nowrap;
}
.help-icon { opacity: .45; cursor: help; flex-shrink: 0; }

.config-input-col { display: flex; align-items: center; flex-shrink: 0; }

/* ── status row ── */
.status-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.status-row code { font-family: monospace; font-size: 12px; opacity: .7; }

/* ── flash ── */
.flash-progress { display: flex; flex-direction: column; gap: 4px; }
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

/* ── light mode overrides ── */
.light-mode .log-block,
.light-mode .flash-log      { background: rgba(0,0,0,.06); color: #111; }
.light-mode .build-terminal { background: #1f2937; }
.light-mode .config-name-badge { background: rgba(0,0,0,.06); }
.light-mode .config-row { border-bottom-color: rgba(0,0,0,.07); }
</style>
