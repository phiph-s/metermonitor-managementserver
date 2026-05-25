<template>
  <br>
  <div class="settings-view">
    <n-spin :show="loading">
      <n-flex>

        <!-- MQTT -->
        <n-card size="small" class="settings-card">

          <template v-if="!setup" #cover>
            <div class="card-type-title" :style="{ '--type-color': '#7c4991' }">
              <span class="type-label">
                <n-icon size="9"><WifiOutlined /></n-icon>
                MQTT
              </span>
            </div>
          </template>

          <div class="fields">
            <br>
            <div class="field-row">
              <span class="field-label">Broker</span>
              <n-input v-model:value="draft.mqtt_broker" placeholder="192.168.1.10" style="width: 220px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Port</span>
              <n-input-number v-model:value="draft.mqtt_port" :min="1" :max="65535" style="width: 110px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Topic</span>
              <n-input v-model:value="draft.mqtt_topic" placeholder="MeterMonitor/#" style="width: 220px;" />
            </div>

            <n-divider style="margin: 4px 0" />

            <div class="field-row">
              <span class="field-label">Username</span>
              <n-input v-model:value="draft.mqtt_username" placeholder="optional" style="width: 180px;" />
            </div>
            <div class="field-row">
              <span class="field-label">Password</span>
              <n-input v-model:value="draft.mqtt_password" type="password" show-password-on="click" placeholder="optional" style="width: 180px;" />
            </div>

            <div v-if="supervisorAvailable !== false" class="ha-autofill">
              <n-button size="small" quaternary type="primary" :loading="fetchingSupervisor" @click="fetchSupervisorCreds">
                <template #icon><n-icon><HomeOutlined /></n-icon></template>
                Fill from Home Assistant
              </n-button>
            </div>
          </div>
        </n-card>

        <!-- Retention -->
        <n-card size="small" class="settings-card">

          <template v-if="!setup" #cover>
            <div class="card-type-title" :style="{ '--type-color': '#498691' }">
              <span class="type-label">
                <n-icon size="9"><StorageOutlined /></n-icon>
                Retention
              </span>
            </div>
          </template>

          <div class="fields">
            <br>
            <div class="field-row">
              <div class="field-label-col">
                <span class="field-label">Max history entries</span>
                <span class="field-hint">Accepted readings stored per meter</span>
              </div>
              <n-input-number v-model:value="draft.max_history" :min="10" :max="10000" style="width: 110px;" />
            </div>
            <div class="field-row">
              <div class="field-label-col">
                <span class="field-label">Max evaluations</span>
                <span class="field-hint">Evaluations with digit images stored per meter</span>
              </div>
              <n-input-number v-model:value="draft.max_evals" :min="10" :max="10000" style="width: 110px;" />
            </div>
          </div>
        </n-card>

      </n-flex>
      <!-- Actions -->
      <br>
      <div class="action-row">
          <n-button round :loading="saving" type="primary" @click="save">Save</n-button>
          <n-button round :loading="restarting" @click="saveAndRestart">
            <template #icon><n-icon><RefreshOutlined /></n-icon></template>
            Save &amp; Apply MQTT
          </n-button>
        </div>
    </n-spin>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import {NCard, NDivider, NInput, NInputNumber, NButton, NIcon, NSpin, useMessage, NFlex, NDropdown} from 'naive-ui';
import {HomeOutlined, RefreshOutlined, WifiOutlined, StorageOutlined, MoreVertFilled} from '@vicons/material';
import { apiService } from '@/services/api';

const message = useMessage();

const loading = ref(true);
const saving = ref(false);
const restarting = ref(false);
const fetchingSupervisor = ref(false);
const supervisorAvailable = ref(null);

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
}
.settings-card {
  min-width: 350px;
  width: 30%;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.card-title {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 13px;
  font-weight: 600;
  opacity: 0.85;
}

.fields {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 4px;
}

.field-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 28px;
}

.field-label {
  font-size: 13px;
  font-weight: 500;
  flex-shrink: 0;
}

.field-label-col {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.field-hint {
  font-size: 11px;
  opacity: 0.45;
}

.ha-autofill {
  display: flex;
  justify-content: flex-end;
  margin-top: 2px;
}

.action-row {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 4px;
}


.card-type-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background-color: rgba(125, 125, 125, 0.1);
  padding: 2px 6px 2px 0;
  color: var(--type-color);
}

.type-label {
  flex: 1;
  text-align: center;
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.15em;
  padding-left: 24px; /* balance the menu button width */
}


</style>
