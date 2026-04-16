<template>
  <div class="settings-view">
    <n-spin :show="loading">
      <n-flex vertical :size="16">

        <!-- MQTT -->
        <n-card size="small">
          <template #cover>
            <div class="section-title">MQTT</div>
          </template>
          <div class="fields">
            <div class="field-row">
              <span class="field-label">Broker</span>
              <n-input v-model:value="draft.mqtt_broker" placeholder="e.g. 192.168.1.10" style="width: 240px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Port</span>
              <n-input-number v-model:value="draft.mqtt_port" :min="1" :max="65535" style="width: 120px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Topic</span>
              <n-input v-model:value="draft.mqtt_topic" placeholder="e.g. MeterMonitor/#" style="width: 240px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Username</span>
              <n-input v-model:value="draft.mqtt_username" placeholder="optional" style="width: 200px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Password</span>
              <n-input v-model:value="draft.mqtt_password" type="password" show-password-on="click" placeholder="optional" style="width: 200px;" />
            </div>
            <div v-if="supervisorAvailable !== false" class="field-row">
              <span class="field-label">
                Auto-fill from HA
                <n-tooltip trigger="hover">
                  <template #trigger>
                    <n-icon size="14" style="cursor: help; opacity: 0.6; margin-left: 4px;"><HelpOutlineOutlined /></n-icon>
                  </template>
                  Fetch MQTT credentials from the Home Assistant Supervisor (only available when running as HA addon)
                </n-tooltip>
              </span>
              <n-button size="small" :loading="fetchingSupervisor" @click="fetchSupervisorCreds">
                <template #icon><n-icon><HomeOutlined /></n-icon></template>
                Use HA MQTT credentials
              </n-button>
            </div>
          </div>
        </n-card>

        <!-- Retention -->
        <n-card size="small">
          <template #cover>
            <div class="section-title">Retention</div>
          </template>
          <div class="fields">
            <div class="field-row">
              <span class="field-label">
                Max detailed history entries
                <n-tooltip trigger="hover">
                  <template #trigger>
                    <n-icon size="14" style="cursor: help; opacity: 0.6; margin-left: 4px;"><HelpOutlineOutlined /></n-icon>
                  </template>
                  Maximum number of accepted meter readings stored per meter
                </n-tooltip>
              </span>
              <n-input-number v-model:value="draft.max_history" :min="10" :max="10000" style="width: 120px;" />
            </div>
            <div class="field-row">
              <span class="field-label">
                Max evaluations
                <n-tooltip trigger="hover">
                  <template #trigger>
                    <n-icon size="14" style="cursor: help; opacity: 0.6; margin-left: 4px;"><HelpOutlineOutlined /></n-icon>
                  </template>
                  Maximum number of evaluations (with digit images) stored per meter
                </n-tooltip>
              </span>
              <n-input-number v-model:value="draft.max_evals" :min="10" :max="10000" style="width: 120px;" />
            </div>
          </div>
        </n-card>

        <!-- Actions -->
        <n-flex justify="flex-end" :size="8">
          <n-button round :loading="saving" type="primary" @click="save">Save</n-button>
          <n-button round :loading="restarting" @click="saveAndRestart">
            <template #icon><n-icon><RefreshOutlined /></n-icon></template>
            Save &amp; Apply MQTT
          </n-button>
        </n-flex>

      </n-flex>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { NCard, NFlex, NInput, NInputNumber, NButton, NIcon, NTooltip, NSpin, useMessage } from 'naive-ui';
import { HomeOutlined, HelpOutlineOutlined, RefreshOutlined } from '@vicons/material';
import { apiService } from '@/services/api';

const message = useMessage();

const loading = ref(true);
const saving = ref(false);
const restarting = ref(false);
const fetchingSupervisor = ref(false);
const supervisorAvailable = ref(null); // null = unknown, true = available, false = not available

const draft = ref({
  mqtt_broker: '',
  mqtt_port: 1883,
  mqtt_username: '',
  mqtt_password: '',
  mqtt_topic: 'MeterMonitor/#',
  max_history: 200,
  max_evals: 100,
});

const load = async () => {
  loading.value = true;
  try {
    const data = await apiService.getJson('api/global-settings');
    draft.value = {
      mqtt_broker: data.mqtt_broker || '',
      mqtt_port: data.mqtt_port || 1883,
      mqtt_username: data.mqtt_username || '',
      mqtt_password: data.mqtt_password || '',
      mqtt_topic: data.mqtt_topic || 'MeterMonitor/#',
      max_history: data.max_history ?? 200,
      max_evals: data.max_evals ?? 100,
    };
  } catch (e) {
    message.error('Failed to load settings');
  } finally {
    loading.value = false;
  }
};

const fetchSupervisorCreds = async () => {
  fetchingSupervisor.value = true;
  try {
    const data = await apiService.getJson('api/global-settings/mqtt-supervisor');
    if (data.mqtt_broker) draft.value.mqtt_broker = data.mqtt_broker;
    if (data.mqtt_port) draft.value.mqtt_port = data.mqtt_port;
    if (data.mqtt_username) draft.value.mqtt_username = data.mqtt_username;
    if (data.mqtt_password) draft.value.mqtt_password = data.mqtt_password;
    supervisorAvailable.value = true;
    message.success('Credentials loaded from Home Assistant');
  } catch (e) {
    supervisorAvailable.value = false;
    message.error('HA Supervisor not available');
  } finally {
    fetchingSupervisor.value = false;
  }
};

const buildPayload = () => ({
  mqtt_broker: draft.value.mqtt_broker || null,
  mqtt_port: draft.value.mqtt_port,
  mqtt_username: draft.value.mqtt_username || null,
  mqtt_password: draft.value.mqtt_password || null,
  mqtt_topic: draft.value.mqtt_topic || null,
  max_history: draft.value.max_history,
  max_evals: draft.value.max_evals,
});

const save = async () => {
  saving.value = true;
  try {
    await apiService.putJson('api/global-settings', buildPayload());
    message.success('Settings saved');
  } catch (e) {
    message.error('Failed to save settings');
  } finally {
    saving.value = false;
  }
};

const saveAndRestart = async () => {
  restarting.value = true;
  try {
    await apiService.putJson('api/global-settings', buildPayload());
    await apiService.postJson('api/global-settings/mqtt/restart', {});
    message.success('Settings saved and MQTT restarted');
  } catch (e) {
    message.error('Failed to apply settings');
  } finally {
    restarting.value = false;
  }
};

onMounted(load);
</script>

<style scoped>
.settings-view {
  max-width: 600px;
}

.section-title {
  text-transform: uppercase;
  width: 100%;
  background-color: rgba(125, 125, 125, 0.1);
  text-align: center;
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.1em;
  padding: 4px 0;
}

.fields {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 14px;
}

.field-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.field-label {
  display: flex;
  align-items: center;
  font-size: 14px;
  flex-shrink: 0;
}
</style>
