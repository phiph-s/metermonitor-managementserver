<template>
  <AddSourceDialog v-model:show="showAddSource" :config="config" @created="getData" />
  <br>

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
              <a href="https://metermonitor-io.github.io/#/" target="_blank" rel="noreferrer" style="text-decoration: none; color: inherit;">
                GitHub Documentation
              </a>
            </n-button>
          </n-flex>
        </div>
        <AddMeterButton @click-add="showAddSource = true" @click-flash="router.push('/flash')"/>
      </n-flex>
      <n-divider />
      <MQTTSetupHelper :config="config"/>
    </n-space>
  </div>
  <template v-if="discoveredMeters.length > 0">
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
    <br>
  </template>
  <div v-if="waterMeters.length > 0" class="filter-row">
    <n-button
      v-for="pill in filterPills"
      :key="pill.value"
      :type="activeFilter === pill.value ? 'primary' : 'default'"
      size="small"
      round
      @click="activeFilter = pill.value"
    >
      <template v-if="pill.color" #icon>
        <span :style="{ color: pill.color, fontSize: '10px' }">●</span>
      </template>
      {{ pill.label }}
    </n-button>
  </div>

  <template v-if="filteredWaterMeters.length > 0">
    <n-flex class="watermeters-row">
      <WaterMeterCard
          v-for="item in filteredWaterMeters"
          :key="item[0]"
          :last_updated="item[1]"
          :meter_name="item[0]"
          :setup="false"
          :rssi="item[2]"
          :last_digits="item[4]"
          :last_result="item[3]"
          :has_bbox="item[5]"
          :decimals="item[6]"
          :meter_type="item[7]"
          :unit="item[8]"
          :last_error="item[9]"
          :source_type="item[10]"
          @removed="getData"
      />
      <AddMeterButton @click-add="showAddSource = true" @click-flash="router.push('/flash')"/>
    </n-flex>
  </template>
  <n-flex class="watermeters-row" v-else-if="discoveredMeters.length !== 0 && waterMeters.length === 0 && config">
    <AddMeterButton @click-add="showAddSource = true" @click-flash="router.push('/flash')"/>
  </n-flex>

</template>

<script setup>
import {computed, onMounted, onUnmounted, ref, watch} from 'vue';
import {NButton, NDivider, NFlex, NH2, NIcon, NSpace} from 'naive-ui';
import router from "@/router";
import WaterMeterCard from "@/components/WaterMeterCard.vue";
import AddSourceDialog from "@/components/AddSourceDialog.vue";
import {useHeaderControls} from '@/composables/headerControls';
import {PendingActionsOutlined} from '@vicons/material';
import MQTTSetupHelper from "@/views/MQTTSetupHelper.vue";
import {METER_TYPES, meterTypeColors, meterTypeLabels} from '@/utils/meterTypeMeta';
import AddMeterButton from "@/views/AddMeterButton.vue";

const discoveredMeters = ref([]);
const waterMeters = ref([]);
const sources = ref([]);
const loading = ref(false);
const config = ref(null);
const showAddSource = ref(false);
const activeFilter = ref('ALL');
const headerControls = useHeaderControls();

const filterPills = computed(() => {
  const all = { label: 'All', value: 'ALL', color: null };
  const typesPresent = new Set(waterMeters.value.map(m => m[7] || 'WATER'));
  const pills = [all];
  for (const t of METER_TYPES) {
    if (typesPresent.has(t)) {
      pills.push({ label: meterTypeLabels[t], value: t, color: meterTypeColors[t] });
    }
  }
  return pills;
});

const filteredWaterMeters = computed(() => {
  if (activeFilter.value === 'ALL') return waterMeters.value;
  return waterMeters.value.filter(m => (m[7] || 'WATER') === activeFilter.value);
});
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

  // Add last_error and source_type from sources to watermeters
  // Backend tuple: [name(0), timestamp(1), rssi(2), result(3), th_digits(4), has_bbox(5), decimals(6), meter_type(7), unit(8)]
  // After JS merge: [..., last_error(9), source_type(10)]
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
  margin-left: 8px;
}

.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
  margin-bottom: 4px;
  align-items: center;
}

.watermeters-row {
  margin-top: 12px;
}

.add-card {
  width: 300px;
  height: 286px;
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

</style>
