<template>
  <n-modal v-model:show="show" preset="card" title="Add meter" style="max-width: 860px;" :mask-closable="!saving && !testing" :closable="!testing">
    <n-space vertical size="large">

      <n-form :model="form" :disabled="saving || testing" label-placement="top">
        <n-grid :cols="24" :x-gap="12" :y-gap="8">
          <n-form-item-gi :span="24" label="Source type">
          <n-select v-model:value="selectedType" :options="typeOptions" data-testid="source-type-select" />
          </n-form-item-gi>

          <n-form-item-gi :span="12" label="Meter name" v-if="selectedType !== 'mqtt'">
            <n-input v-model:value="form.name" placeholder="e.g. Hauptzaehler" data-testid="source-name-input" />
          </n-form-item-gi>

          <n-form-item-gi :span="6" label="Enabled" v-if="selectedType !== 'mqtt'">
            <n-switch v-model:value="form.enabled" />
          </n-form-item-gi>

          <n-form-item-gi :span="6" label="Poll interval (m)" v-if="selectedType !== 'mqtt'">
            <n-input-number v-model:value="form.poll_interval_m" :min="1" :max="3600" data-testid="source-poll-interval" />
          </n-form-item-gi>

          <!-- Home Assistant camera config -->
          <template v-if="selectedType === 'ha_camera'">
            <n-form-item-gi :span="24" label="Camera entity">
              <n-select
                v-model:value="form.camera_entity_id"
                :options="cameraOptions"
                filterable
                placeholder="Select a Home Assistant camera.* entity"
                :loading="loadingCameras"
                data-testid="ha-camera-select"
              />
            </n-form-item-gi>

            <n-form-item-gi :span="18" label="Flash light entity (we recommend to use a always-on LED instead)">
              <n-space vertical style="width: 100%;">
                <n-space align="center" style="width: 100%;" :wrap="false">
                  <n-switch v-model:value="form.flash_enabled" />
                  <n-input
                    v-model:value="form.flash_entity_id"
                    placeholder="e.g. light.watermeter_led"
                    :disabled="!form.flash_enabled"
                  />
                  <n-button
                    secondary
                    :disabled="!suggestedFlash || !form.flash_enabled"
                    @click="useSuggestedFlash"
                  >
                    Use suggested
                  </n-button>
                </n-space>
                <div style="opacity: 0.7;" v-if="suggestedFlash && form.flash_enabled">
                  Suggested: <span class="code-inline">{{ suggestedFlash }}</span>
                </div>
              </n-space>
            </n-form-item-gi>

            <n-form-item-gi :span="6" label="Flash delay (ms)">
              <n-input-number v-model:value="form.flash_delay_ms" :min="0" :max="10000" :disabled="!form.flash_enabled" />
            </n-form-item-gi>

            <n-form-item-gi :span="24" label="Test / preview">
              <n-space vertical>
                <n-space>
                  <n-button
                    type="primary"
                    :loading="testing"
                    @click="testCapture"
                    :disabled="!form.camera_entity_id"
                  >
                    Test capture
                  </n-button>

                  <n-text depth="3" v-if="testHint">
                    {{ testHint }}
                  </n-text>
                </n-space>

                <n-progress v-if="testing && form.flash_delay_ms > 0" :percentage="progress" />

                <div v-if="previewBbox" class="preview">
                  <img :src="previewBbox" alt="Preview" />
                </div>
              </n-space>
            </n-form-item-gi>
          </template>

          <!-- HTTP source config -->
          <template v-if="selectedType === 'http'">
            <n-form-item-gi :span="24" label="Image URL">
            <n-input v-model:value="form.http_url" placeholder="https://example.com/camera.jpg" data-testid="http-url-input" />
            </n-form-item-gi>

            <n-form-item-gi :span="12" label="Headers (JSON)">
              <n-input
                v-model:value="form.http_headers"
                type="textarea"
                :autosize="{ minRows: 3, maxRows: 6 }"
                placeholder='{"Authorization": "Bearer ..."}'
              />
            </n-form-item-gi>

            <n-form-item-gi :span="12" label="Body (optional)">
              <n-input
                v-model:value="form.http_body"
                type="textarea"
                :autosize="{ minRows: 3, maxRows: 6 }"
                placeholder="Optional body (sent with GET)"
              />
            </n-form-item-gi>

            <n-form-item-gi :span="24" label="Test / preview">
              <n-space vertical>
                <n-space>
                  <n-button
                    type="primary"
                    :loading="testing"
                    @click="testCapture"
                    :disabled="!form.http_url"
                  >
                    Test capture
                  </n-button>

                  <n-text depth="3" v-if="testHint">
                    {{ testHint }}
                  </n-text>
                </n-space>

                <div v-if="previewBbox" class="preview">
                  <img :src="previewBbox" alt="Preview" />
                </div>
              </n-space>
            </n-form-item-gi>
          </template>
        </n-grid>

        <div v-if="selectedType === 'mqtt'">
          <n-alert type="info" >
            MQTT sources don’t need manual setup.
            <br />
            When a device publishes images to the configured broker/topic, it will appear under “Waiting for setup”.
          </n-alert><br>
          <MQTTSetupHelper :config="config"/>
        </div>

      </n-form>


      <n-space justify="end">
        <n-button @click="close" :disabled="saving || testing">Cancel</n-button>
        <n-button type="success" @click="create" :loading="saving" :disabled="!canCreate || testing" data-testid="create-source-button">
          Create source
        </n-button>
      </n-space>
    </n-space>
  </n-modal>
</template>

<script setup>
/* global defineProps, defineEmits */
import { computed, onMounted, ref, watch } from 'vue';
import {
  NModal,
  NSpace,
  NForm,
  NGrid,
  NFormItemGi,
  NSelect,
  NInput,
  NInputNumber,
  NSwitch,
  NButton,
  NAlert,
  NText,
  NProgress, useMessage,
} from 'naive-ui';
import { apiService } from '@/services/api';
import MQTTSetupHelper from "@/views/MQTTSetupHelper.vue";

const props = defineProps({
  show: { type: Boolean, default: false },
  config: { type: Object, default: null },
});

const emit = defineEmits(['update:show', 'created']);

const show = computed({
  get: () => props.show,
  set: (v) => emit('update:show', v),
});

const selectedType = ref('ha_camera');

const typeOptions = [
  { label: 'Home Assistant (Camera entity)', value: 'ha_camera' },
  { label: 'MQTT (auto-discovery)', value: 'mqtt' },
  { label: 'HTTP (URL)', value: 'http' },
];

const form = ref({
  name: '',
  enabled: true,
  poll_interval_m: 10,
  camera_entity_id: null,
  flash_enabled: false,
  flash_entity_id: null,
  flash_delay_ms: 10000,
  http_url: '',
  http_headers: '',
  http_body: '',
});

const saving = ref(false);
const testing = ref(false);
const loadingCameras = ref(false);
const cameras = ref([]);
const previewBbox = ref(null);
const testHint = ref('');
const progress = ref(0);
let progressInterval = null;

const cameraOptions = computed(() =>
  cameras.value.map((c) => ({ label: c.name || c.entity_id, value: c.entity_id, raw: c }))
);

const suggestedFlash = computed(() => {
  const cam = cameras.value.find((c) => c.entity_id === form.value.camera_entity_id);
  return cam?.suggested_flash_entity_id || null;
});

function useSuggestedFlash() {
  if (suggestedFlash.value) {
    form.value.flash_entity_id = suggestedFlash.value;
  }
}

const canCreate = computed(() => {
  if (!form.value.name?.trim()) return false;
  if (selectedType.value === 'mqtt') return false;
  if (!form.value.poll_interval_m || form.value.poll_interval_m < 1) return false;
  if (selectedType.value === 'ha_camera') {
    return !!form.value.camera_entity_id;
  }
  if (selectedType.value === 'http') {
    return !!form.value.http_url?.trim();
  }
  return true;
});

const parseHeaders = (raw) => {
  if (!raw || !raw.trim()) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch (e) {
    return { __invalid__: true };
  }
  return { __invalid__: true };
};

async function loadCameras() {
  if (selectedType.value !== 'ha_camera') return;
  loadingCameras.value = true;
  try {
    const r = await apiService.getJson('api/ha/cameras');
    cameras.value = r.cameras || [];
  } catch (e) {
    console.error('Failed to load HA cameras', e);
    cameras.value = [];
  } finally {
    loadingCameras.value = false;
  }
}

watch(
  () => selectedType.value,
  async () => {
    previewBbox.value = null;
    testHint.value = '';
    if (selectedType.value === 'ha_camera') await loadCameras();
  }
);

watch(
  () => form.value.camera_entity_id,
  () => {
    // auto-prefill flash if suggested
    const cam = cameras.value.find((c) => c.entity_id === form.value.camera_entity_id);
    if (cam?.suggested_flash_entity_id) {
      form.value.flash_entity_id = cam.suggested_flash_entity_id;
      form.value.flash_enabled = true;
    }
  }
);

function close() {
  show.value = false;
}

const message = useMessage();

async function create() {
  saving.value = true;
  try {
    const payload = {
      name: form.value.name.trim(),
      source_type: selectedType.value,
      enabled: !!form.value.enabled,
      poll_interval_s: selectedType.value === 'mqtt' ? null : form.value.poll_interval_m * 60,
      config: undefined,
    };

    if (selectedType.value === 'ha_camera') {
      payload.config = {
        camera_entity_id: form.value.camera_entity_id,
        flash_entity_id: form.value.flash_enabled ? (form.value.flash_entity_id || null) : '',
        flash_delay_ms: form.value.flash_delay_ms,
      };
    }
    if (selectedType.value === 'http') {
      const headers = parseHeaders(form.value.http_headers);
      if (headers && headers.__invalid__) {
        message.error('Headers must be valid JSON object (e.g. {"Authorization": "Bearer ..."}).');
        return;
      }
      payload.config = {
        url: form.value.http_url.trim(),
        headers: headers || null,
        body: form.value.http_body?.trim() || null,
      };
    }

    await apiService.postJson('api/sources', payload);
    emit('created');
    show.value = false;

    //clear form
    form.value = {
      name: '',
      enabled: true,
      poll_interval_m: 10,
      camera_entity_id: null,
      flash_enabled: true,
      flash_entity_id: null,
      flash_delay_ms: 10000,
      http_url: '',
      http_headers: '',
      http_body: '',
    };
  } catch (e) {
    console.error('Create source failed', e);
  } finally {
    saving.value = false;
  }
}

async function testCapture() {
  testing.value = true;
  testHint.value = '';
  previewBbox.value = null;
  progress.value = 0;

  if (selectedType.value === 'ha_camera' && form.value.flash_delay_ms > 0) {
    const totalTime = form.value.flash_delay_ms;
    const step = 100; // update every 100ms
    const increment = (step / totalTime) * 100;
    progressInterval = setInterval(() => {
      progress.value = Math.min(progress.value + increment, 100);
    }, step);
  }

  try {
    let r;
    if (selectedType.value === 'http') {
      const headers = parseHeaders(form.value.http_headers);
      if (headers && headers.__invalid__) {
        message.error('Headers must be valid JSON object (e.g. {"Authorization": "Bearer ..."}).');
        return;
      }
      r = await apiService.postJson(`api/capture-now`, {
        http_url: form.value.http_url.trim(),
        http_headers: headers || null,
        http_body: form.value.http_body?.trim() || null,
      });
    } else {
      r = await apiService.postJson(`api/capture-now`, {
        cam_entity_id: form.value.camera_entity_id,
        flash_entity_id: form.value.flash_enabled ? (form.value.flash_entity_id || null) : '',
        flash_delay_ms: form.value.flash_delay_ms,
      });
    }

    if (!r.result) {
      testHint.value = r.message || 'Capture failed.';
      return;
    }

    testHint.value = selectedType.value === 'http'
      ? 'Capture ok (http)'
      : `Capture ok (flash: ${r.flash_used ? 'on' : 'off'})`;
    previewBbox.value = `data:image/${r.format};base64,${r.data}`;
  } catch (e) {
    console.error('Test capture failed', e);
    message.error('Test capture failed: ' + (e.message || e), {
      closable: true,
      duration: 60000,
    });
  } finally {
    testing.value = false;
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    progress.value = 0;
  }
}

onMounted(async () => {
  await loadCameras();
});
</script>

<style scoped>
.code-inline{
  font-family: monospace, monospace;
  padding: 2px 6px;
  border-radius: 6px;
  background-color: rgba(255,255,255,0.08);
}

.preview {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 10px;
  max-width: 760px;
}
.preview img {
  width: 100%;
  display: block;
  border-radius: 8px;
}
</style>
