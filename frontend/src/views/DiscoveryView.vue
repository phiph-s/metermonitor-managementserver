<template>
  <AddSourceDialog v-model:show="showAddSource" @created="getData" />

  <div v-if="discoveredMeters.length === 0 && waterMeters.length === 0 && config">
    <n-space vertical size="large">
      <n-flex>
        <div>
          <br>
          <n-h2>Welcome to MeterMonitor!</n-h2>
          <div>
            If you use MQTT, devices will appear automatically once they publish images.<br>
            You can also add a Home Assistant camera source via the button above.
          </div>
          <n-flex>
            <n-button quaternary type="info">
              <a href="https://esphome.io/components/camera/esp32_camera.html" target="_blank" rel="noreferrer" style="text-decoration: none; color: inherit;">
                ESPHome ESP32-CAM setup guide
              </a>
            </n-button>

            <n-button quaternary type="info">
              <a href="https://github.com/phiph-s/metermonitor-managementserver/" target="_blank" rel="noreferrer" style="text-decoration: none; color: inherit;">
                GitHub Documentation
              </a>
            </n-button>
          </n-flex>
        </div>
        <div class="add-card" data-testid="add-watermeter-card" @click="showAddSource = true" style="margin-left: 20px;">
          <div class="add-card-inner">
            <n-icon><AddOutlined /></n-icon>
            <span>Add watermeter</span>
          </div>
        </div>
      </n-flex>
      <n-divider />
      <div>
        <b>MQTT payload format</b>
        <div class="code">
          {<br>
          &nbsp;"name": "unique name",<br>
          &nbsp;"picture_number": 57,<br>
          &nbsp;"WiFi-RSSI": -57,<br>
          &nbsp;"picture": {<br>
          &nbsp;&nbsp;"format": "jpeg",<br>
          &nbsp;&nbsp;"timestamp": "2025-...",<br>
          &nbsp;&nbsp;"width": 640,<br>
          &nbsp;&nbsp;"height": 320,<br>
          &nbsp;&nbsp;"length": 12345,<br>
          &nbsp;&nbsp;"data": "..."<br>
          &nbsp;}<br>
          }
        </div>
      </div>

      <div>
        <b>Current MQTT config</b><br />
        Broker: <span class="code-inline">{{ config?.mqtt?.broker }}</span><br />
        Topic: <span class="code-inline">{{ config?.mqtt?.topic }}</span>
      </div>
    </n-space>
  </div>

  <template v-if="discoveredMeters.length > 0">
    <div class="elevated-list">
      <div class="elevated-title">
        <n-icon><PendingActionsOutlined /></n-icon>
        <span>Waiting for setup</span>
      </div>
      <n-flex>
        <WaterMeterCard
          v-for="item in discoveredMeters"
          :key="item.id"
          :last_updated="item[1]"
          :meter_name="item[0]"
          :setup="true"
          :rssi="item[2]"
          :source_type="item[3]"
          @removed="getData"
        />
      </n-flex>
    </div>
  </template>

  <n-h2 v-if="waterMeters.length > 0">Watermeters</n-h2>

  <template v-if="waterMeters.length > 0">
    <n-flex class="watermeters-row">
      <WaterMeterCard
          v-for="item in waterMeters"
          :key="item.id"
          :last_updated="item[1]"
          :meter_name="item[0]"
          :setup="false"
          :rssi="item[2]"
          :last_digits="item[4]"
          :last_result="item[3]"
          :has_bbox="item[5]"
          :decimals="item[6]"
          :last_error="item[7]"
          :source_type="item[8]"
          @removed="getData"
      />
      <div class="add-card" data-testid="add-watermeter-card" @click="showAddSource = true">
        <div class="add-card-inner">
          <n-icon><AddOutlined /></n-icon>
          <span>Add watermeter</span>
        </div>
      </div>
    </n-flex>
  </template>
  <n-flex class="watermeters-row" v-else>
    <div class="add-card" data-testid="add-watermeter-card" @click="showAddSource = true" >
      <div class="add-card-inner">
        <n-icon><AddOutlined /></n-icon>
        <span>Add watermeter</span>
      </div>
    </div>
  </n-flex>

</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { NH2, NFlex, NButton, NDivider, NCard, NSpace, NIcon } from 'naive-ui';
import router from "@/router";
import WaterMeterCard from "@/components/WaterMeterCard.vue";
import AddSourceDialog from "@/components/AddSourceDialog.vue";
import { useHeaderControls } from '@/composables/headerControls';
import { PendingActionsOutlined } from '@vicons/material';
import { AddOutlined } from '@vicons/material';

const discoveredMeters = ref([]);
const waterMeters = ref([]);
const sources = ref([]);
const loading = ref(false);
const config = ref(null);
const showAddSource = ref(false);
const headerControls = useHeaderControls();
let evaluationEventHandler = null;

const host = import.meta.env.VITE_HOST;

// add secret to header of fetch request
const getData = async () => {
  loading.value = true;
  let response = await fetch(host + 'api/discovery', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  if (response.status === 401) {
    router.push({ path: '/unlock' });
  }
  discoveredMeters.value = (await response.json())["watermeters"];

  response = await fetch(host + 'api/watermeters', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  waterMeters.value = (await response.json())["watermeters"];

  // Load sources to get last_error
  response = await fetch(host + 'api/sources', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  const sourcesData = (await response.json())["sources"];
  sources.value = sourcesData;

  // Add last_error from sources to watermeters
  waterMeters.value = waterMeters.value.map(meter => {
    const source = sourcesData.find(s => s.name === meter[0]);
    return [...meter, source?.last_error || null, source?.source_type || 'mqtt'];
  });

  loading.value = false;

  response = await fetch(host + 'api/config', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  config.value = await response.json();
}

onMounted(() => {
  getData();
  evaluationEventHandler = () => {
    getData();
  };
  window.addEventListener('meter-evaluation-updated', evaluationEventHandler);
  if (headerControls) {
    headerControls.setHeader({
      showRefresh: true,
      onRefresh: getData,
      refreshLoading: loading.value
    });
  }
});

onUnmounted(() => {
  if (evaluationEventHandler) {
    window.removeEventListener('meter-evaluation-updated', evaluationEventHandler);
  }
  if (headerControls) {
    headerControls.resetHeader();
  }
});

watch(loading, (next) => {
  if (!headerControls) return;
  headerControls.setHeader({ refreshLoading: next });
});

</script>

<style scoped>
.elevated-list {
  padding: 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
}

.light-mode .elevated-list {
  background: rgba(0, 0, 0, 0.04);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.06);
}

.elevated-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  margin-bottom: 8px;
}

.watermeters-row {
  margin-top: 18px;
}

.add-card {
  width: 300px;
  height: 275px;
  min-height: 180px;
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  color: rgba(255, 255, 255, 0.7);
}

.add-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255, 255, 255, 0.5);
  color: rgba(255, 255, 255, 0.9);
}

.light-mode .add-card {
  border-color: rgba(0, 0, 0, 0.2);
  color: rgba(0, 0, 0, 0.6);
}

.light-mode .add-card:hover {
  border-color: rgba(0, 0, 0, 0.4);
  color: rgba(0, 0, 0, 0.8);
}

.add-card-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 16px;
}

.code{
  font-family: monospace, monospace;
  background-color: rgba(255,255,255,0.1)
}

div.code{
  background: none;
}

.code-inline{
  font-family: monospace, monospace;
  padding: 2px 6px;
  border-radius: 6px;
  background-color: rgba(255,255,255,0.08);
}
</style>
