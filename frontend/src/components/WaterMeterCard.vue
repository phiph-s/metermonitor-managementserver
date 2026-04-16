<template>
  <n-flex vertical>
    <n-card size="small" class="meter-card" :class="{ 'state-error': hasError, 'state-warning': !hasBB && !setup }" :style="setup?'width: 375px;':''">
      <template #header>
        <div class="card-header">
          <div class="title-group">
            <span class="card-title" :title="meter_name">{{ meter_name }}</span>
            <span class="source-pill" :style="{ '--pill-color': sourceColor }">
              <n-icon size="14"><component :is="sourceIcon" /></n-icon>
              <span>{{ sourceLabel }}</span>
            </span>
          </div>
          <div class="header-meta">
            <span class="timestamp">{{ last_updated_locale }}</span>
            <n-dropdown :options="menuOptions" @select="handleMenuSelect">
              <n-button text>
                <template #icon>
                  <n-icon><MoreVertFilled /></n-icon>
                </template>
              </n-button>
            </n-dropdown>
          </div>
        </div>
      </template>

      <SourceCollapse v-if="setup" :source="source"></SourceCollapse>

      <div class="digits-row" v-if="last_digits">
        <img
          v-for="[i, base64] in last_digits.entries()"
          :key="i + 'c'"
          class="digit theme-revert"
          :style="`width:calc(160px / ${last_digits.length});`"
          :src="'data:image/png;base64,' + base64"
          :alt="meter_name"
        />
        <span class="digit" :style="`width:calc(160px / ${last_digits.length});`"></span>
      </div>

      <div class="result-row" v-if="last_result && last_digits">
        <span
          class="prediction google-sans-code"
          v-for="[i, digit] in (last_result + '').padStart(last_digits.length, '0').split('').entries()"
          :key="i + 'd'"
          :class="{ faded: isDecimalDigitIndex(i, last_digits.length) }"
        >
          <template v-if="isDecimalSeparatorIndex(i, last_digits.length)">
            {{ digit }},
          </template>
          <template v-else>
            {{ digit }}
          </template>
        </span>
        <span class="unit">m³</span>
      </div>

      <div v-if="historyData.length > 1" class="sparkline-container">
        <div class="sparkline-labels">
          <span>{{ firstValue }}</span>
          <span>{{ lastValue }}</span>
        </div>
        <apexchart
          type="area"
          height="42"
          width="100%"
          :options="chartOptions"
          :series="chartSeries"
        />
      </div>

      <template #action>
        <div class="card-footer">
          <router-link v-if="setup" :to="'/setup/'+meter_name">
            <n-button round size="small">Setup</n-button>
          </router-link>
          <router-link v-else :to="'/meter/'+meter_name">
            <n-button round size="small">View</n-button>
          </router-link>
          <WifiStatus v-if="rssi" :rssi="rssi" />
        </div>
      </template>
    </n-card>
    <div v-if="hasError" class="card-note error">
      {{ last_error }}
    </div>
    <div v-if="!hasBB && !setup" class="card-note warning">
      No bounding box found in the last capture
    </div>
  </n-flex>
</template>

<script setup>
import {NCard, NButton, NFlex, NDropdown, NIcon, useDialog} from 'naive-ui';
import {defineProps, computed, ref, onMounted, watch} from 'vue';
import { MoreVertFilled, HomeOutlined, PublicFilled, WifiTetheringOutlined, HelpOutlineOutlined } from '@vicons/material';
import WifiStatus from "@/components/WifiStatus.vue";
import { useThemeStore } from '@/stores/themeStore';
import { storeToRefs } from 'pinia';
import { getSourceColor, getSourceLabel, normalizeSourceType } from '@/utils/sourceMeta';
import SourceCollapse from "@/components/SourceCollapse.vue";
import { apiService } from '@/services/api';

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const props = defineProps([
    'meter_name',
    'last_updated', // eg "2025-02-04T03:15:31"
    'setup',
    'last_digits',
    'last_result',
    'rssi',
    'last_error',
    'has_bbox',
    'source_type',
    'decimals'
]);

const hasError = computed(() => !!props.last_error);
const hasBB = computed(() => !!props.has_bbox);
const sourceType = computed(() => normalizeSourceType(props.source_type));
const sourceColor = computed(() => getSourceColor(sourceType.value));
const sourceLabel = computed(() => getSourceLabel(sourceType.value));
const sourceIcon = computed(() => {
  if (sourceType.value === 'mqtt') return WifiTetheringOutlined;
  if (sourceType.value === 'ha_camera') return HomeOutlined;
  if (sourceType.value === 'http') return PublicFilled;
  return HelpOutlineOutlined;
});

const historyData = ref([]);
const source = ref(null);
const host = import.meta.env.VITE_HOST;
const dialog = useDialog();

const emit = defineEmits(['removed']);

const menuOptions = [
  { label: 'Remove', key: 'remove' }
];

const handleMenuSelect = (key) => {
  if (key === 'remove') {
    dialog.warning({
      title: 'Confirm Removal',
      content: 'Are you sure you want to remove this meter and its source?',
      positiveText: 'Remove',
      negativeText: 'Cancel',
      onPositiveClick: removeMeter
    });
  }
};

const removeMeter = async () => {
  try {
    const response = await fetch(`${host}api/watermeters/${props.meter_name}`, {
      method: 'DELETE',
      headers: { 'secret': localStorage.getItem('secret') }
    });
    if (response.ok) {
      emit('removed');
    } else {
      console.error('Failed to remove meter');
    }
  } catch (e) {
    console.error('Error removing meter:', e);
  }
};

const loadHistory = async () => {
  if (props.setup) return;
  try {
    const response = await fetch(host + `api/watermeters/${props.meter_name}/history`, {
      headers: { 'secret': localStorage.getItem('secret') }
    });
    if (response.ok) {
      const data = await response.json();
      // Sort by timestamp and take last 20 entries
      historyData.value = (data.history || [])
        .sort((a, b) => new Date(a[1]) - new Date(b[1]))
        .slice(-20);
    }
  } catch (e) {
    console.error('Failed to load history for card:', e);
  }
};

const loadSource = async () => {
  if (!props.setup) {
    source.value = null;
    return;
  }
  try {
    const data = await apiService.getJson('api/sources');
    source.value = data.sources?.find((item) => item.name === props.meter_name) || null;
  } catch (e) {
    console.error('Failed to load source for card:', e);
    source.value = null;
  }
};

onMounted(() => {
  loadHistory();
  loadSource();
});

watch(() => [props.meter_name, props.setup], () => {
  loadSource();
});

const last_updated_locale = computed(() => {
  if (!props.last_updated) return '';
  const date = new Date(props.last_updated);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' · ' + date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
});

const meterDecimals = computed(() => {
  const raw = Number.isFinite(props.decimals) ? props.decimals : 3;
  return Math.max(0, raw);
});

const meterScale = computed(() => 10 ** meterDecimals.value);

const isDecimalSeparatorIndex = (idx, digitLength) => {
  const decimals = Math.min(meterDecimals.value, digitLength);
  if (decimals === 0) return false;
  return idx === digitLength - decimals - 1;
};

const isDecimalDigitIndex = (idx, digitLength) => {
  const decimals = Math.min(meterDecimals.value, digitLength);
  if (decimals === 0) return false;
  return idx >= digitLength - decimals;
};

const firstValue = computed(() => {
  if (historyData.value.length === 0) return '';
  return (historyData.value[0][0] / meterScale.value).toFixed(Math.min(meterDecimals.value + 1, 4));
});

const lastValue = computed(() => {
  if (historyData.value.length === 0) return '';
  return (historyData.value[historyData.value.length - 1][0] / meterScale.value).toFixed(Math.min(meterDecimals.value + 1, 4));
});

const chartSeries = computed(() => [{
  name: 'Value',
  data: historyData.value.map(item => ({
    x: new Date(item[1]).getTime(),
    y: item[0] / meterScale.value
  }))
}]);

const chartOptions = computed(() => ({
  chart: {
    type: 'area',
    sparkline: { enabled: true },
    animations: { enabled: false },
    background: 'transparent',
  },
  stroke: {
    curve: 'smooth',
    width: 2,
  },
  fill: {
    type: 'gradient',
    gradient: {
      shadeIntensity: 1,
      opacityFrom: 0.4,
      opacityTo: 0.1,
    }
  },
  colors: [isDark.value ? '#18bcf2' : '#0891b2'],
  tooltip: {
    enabled: true,
    theme: isDark.value ? 'dark' : 'light',
    x: {
      format: 'dd MMM HH:mm'
    },
    y: {
      formatter: (val) => val.toFixed(Math.min(meterDecimals.value + 1, 4)) + ' m³'
    }
  },
  xaxis: {
    type: 'datetime',
  },
  yaxis: {
    show: false,
    min: 0,
    max: 100,
  },
  grid: {
    show: false,
    padding: { left: 0, right: 0, top: 0, bottom: 0 }
  },
}));
</script>

<style scoped>
.meter-card {
  width: 300px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.06);
  overflow: hidden;
}

.light-mode .meter-card {
  background: rgba(0, 0, 0, 0.02);
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.state-warning {
  border-color: rgba(240, 138, 0, 0.4);
}

.state-error {
  border-color: rgba(208, 48, 80, 0.5);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.title-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.card-title {
  display: block;
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.source-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--pill-color);
}

.light-mode .source-pill {
  background: rgba(0, 0, 0, 0.06);
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.timestamp {
  font-size: 11px;
  opacity: 0.6;
}

.digits-row {
  display: flex;
  justify-content: space-around;
  margin-top: 6px;
  margin-bottom: 4px;
}

.digit {
  margin: 3px;
  height: 40px;
  mix-blend-mode: screen;
  opacity: 0.7;
  border-radius: 3px;
}

.result-row {
  display: flex;
  align-items: baseline;
  justify-content: space-evenly;
  margin: 4px 0 0 0;
  gap: 4px;
}

.prediction {
  width: 16px;
  text-wrap: nowrap;
  font-size: 1.4em;
}

.prediction.faded {
  color: rgba(255, 154, 154, 0.9);
}

.unit {
  font-size: 12px;
  opacity: 0.6;
}

.sparkline-container {
  margin: 8px -16px 0 -16px;
  position: relative;
}

.sparkline-labels {
  display: flex;
  justify-content: space-between;
  padding: 0 8px;
  font-size: 10px;
  opacity: 0.5;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1;
}

.card-footer {
  display: flex;
  align-items: center;
  border-radius: 14px;
  justify-content: space-between;
  gap: 8px;
}

.card-note {
  margin-top: 6px;
  font-size: 12px;
  width: 277px;
  padding: 6px 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
}

.light-mode .card-note {
  background: rgba(0, 0, 0, 0.04);
}

.card-note.error {
  color: #d03050;
  border-left: 3px solid #d03050;
}

.card-note.warning {
  color: rgb(240, 138, 0);
  border-left: 3px solid rgb(240, 138, 0);
}
</style>
