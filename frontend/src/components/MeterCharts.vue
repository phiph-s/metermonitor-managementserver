<template>
  <div class="chart-shell">
    <div class="chart-panel">
      <div class="chart-range">
        <span class="range-label">Usage</span>
        <span class="range-value">{{ formatUsage(usageFrom) }}</span>
        <span class="range-arrow">→</span>
        <span class="range-value">{{ formatUsage(usageTo) }}</span>
        <span class="range-spacer"></span>
        <span class="range-duration">{{ durationLabel }}</span>
        <span class="range-dot">·</span>
        <span class="range-confidence">avg {{ averageConfidence }}</span>
        <span class="range-dot">·</span>
        <span class="range-confidence">median {{ medianConfidence }}</span>
      </div>
      <apexchart
        v-if="combinedSeries.length > 0"
        width="100%"
        height="140"
        type="line"
        :options="combinedChartOptions"
        :series="combinedSeries"
        :key="'chart-' + isDark"
      ></apexchart>
      <div v-else class="chart-empty">
        No history data available
      </div>
    </div>
    <div v-if="combinedSeries.length > 0" class="chart-footer">
      <span>{{ leftTimestamp }}</span>
      <span>{{ midTimestamp }}</span>
      <span>{{ rightTimestamp }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed, defineProps } from 'vue';
import { useThemeStore } from '@/stores/themeStore';
import { storeToRefs } from 'pinia';

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const props = defineProps({
  history: {
    type: Object,
    default: null
  },
  unit: {
    type: String,
    default: 'm³'
  }
});

const sortedHistory = computed(() => {
  if (!props.history || !props.history.history) return [];
  return [...props.history.history].sort((a, b) => new Date(a[1]) - new Date(b[1]));
});

const usageSeries = computed(() => {
  if (sortedHistory.value.length === 0) return [];

  const data = sortedHistory.value.map(item => ({
    x: new Date(item[1]).getTime(),
    y: item[0] / 1000
  }));

  return [{
    name: 'Usage',
    data
  }];
});

const confidenceSeries = computed(() => {
  if (sortedHistory.value.length === 0) return [];

  const data = sortedHistory.value.map(item => ({
    x: new Date(item[1]).getTime(),
    y: item[2] * 100
  }));

  return [{
    name: 'Confidence',
    data
  }];
});

const combinedSeries = computed(() => {
  if (usageSeries.value.length === 0 && confidenceSeries.value.length === 0) return [];
  const usage = usageSeries.value.length
    ? [{ ...usageSeries.value[0], type: 'area' }]
    : [];
  const confidence = confidenceSeries.value.length
    ? [{ ...confidenceSeries.value[0], type: 'column' }]
    : [];
  return [...usage, ...confidence];
});

const lastIndex = computed(() => {
  if (!combinedSeries.value.length) return -1;
  return combinedSeries.value[0]?.data?.length ? combinedSeries.value[0].data.length - 1 : -1;
});

const usageMin = computed(() => {
  if (usageSeries.value.length === 0) return null;
  return Math.min(...usageSeries.value[0].data.map(point => point.y));
});

const usageMax = computed(() => {
  if (usageSeries.value.length === 0) return null;
  return Math.max(...usageSeries.value[0].data.map(point => point.y));
});

const usageFrom = computed(() => {
  if (usageSeries.value.length === 0) return '—';
  return usageSeries.value[0].data[0]?.y ?? '—';
});

const usageTo = computed(() => {
  if (usageSeries.value.length === 0) return '—';
  return usageSeries.value[0].data[usageSeries.value[0].data.length - 1]?.y ?? '—';
});

const durationLabel = computed(() => {
  if (sortedHistory.value.length < 2) return '—';
  const start = new Date(sortedHistory.value[0][1]).getTime();
  const end = new Date(sortedHistory.value[sortedHistory.value.length - 1][1]).getTime();
  const diffMs = Math.max(0, end - start);
  const minutes = Math.floor(diffMs / 60000);
  const days = Math.floor(minutes / 1440);
  const hours = Math.floor((minutes % 1440) / 60);
  const mins = minutes % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
});

const confidenceValues = computed(() => {
  return sortedHistory.value
    .map((item) => item[2])
    .filter((value) => value !== null && value !== undefined)
    .map((value) => Number(value) * 100)
    .filter((value) => !Number.isNaN(value));
});

const averageConfidence = computed(() => {
  if (confidenceValues.value.length === 0) return '—';
  const sum = confidenceValues.value.reduce((acc, value) => acc + value, 0);
  return `${(sum / confidenceValues.value.length).toFixed(1)}%`;
});

const medianConfidence = computed(() => {
  if (confidenceValues.value.length === 0) return '—';
  const sorted = [...confidenceValues.value].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return `${median.toFixed(1)}%`;
});

const leftTimestamp = computed(() => formatShortDate(sortedHistory.value[0]?.[1]));
const rightTimestamp = computed(() => formatShortDate(sortedHistory.value[sortedHistory.value.length - 1]?.[1]));
const midTimestamp = computed(() => {
  if (sortedHistory.value.length === 0) return '—';
  const midIndex = Math.floor(sortedHistory.value.length / 2);
  return formatShortDate(sortedHistory.value[midIndex]?.[1]);
});

const formatShortDate = (value) => {
  if (!value) return '—';
  const date = new Date(value);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short'
  });
};

const formatUsage = (value) => {
  if (value === null || value === undefined || value === '—') return '—';
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return '—';
  return `${numeric.toFixed(2)} ${props.unit}`;
};

const usageColor = computed(() => (isDark.value ? '#22d3ee' : '#0ea5e9'));
const confidenceColor = computed(() => (isDark.value ? '#f59e0b' : '#d97706'));

const combinedChartOptions = computed(() => ({
  theme: { mode: isDark.value ? 'dark' : 'light' },
  chart: {
    type: 'line',
    zoom: { enabled: false },
    background: '#00000000',
    toolbar: { show: false },
    animations: { enabled: true, easing: 'easeinout', speed: 600 },
    sparkline: { enabled: true },
    padding: { top: 18, right: 12, left: 12, bottom: 8 }
  },
  yaxis: [
    {
      title: { text: '' },
      min: usageMin.value,
      max: usageMax.value,
      labels: { formatter: () => "" }
    },
    {
      opposite: true,
      min: 0,
      max: 100,
      title: { text: '' },
      labels: { formatter: (value) => `${Math.round(value)}%` }
    }
  ],
  plotOptions: {
    bar: {
      columnWidth: '75%',
      borderRadius: 4
    }
  },
  stroke: { curve: 'smooth', width: [2.5, 0] },
  fill: {
    type: ['gradient', 'solid'],
    opacity: [0.18, 0.55],
    gradient: {
      shadeIntensity: 0.5,
      opacityFrom: 0.2,
      opacityTo: 0.02,
      stops: [0, 70, 100]
    }
  },
  tooltip: {
    x: { show: false },
    y: [
      { formatter: (value) => `${value.toFixed(3)} ${props.unit}` },
      { formatter: (value) => `${value.toFixed(1)}%` }
    ],
    marker: { show: false }
  },
  colors: [usageColor.value, confidenceColor.value],
  markers: {
    size: 0,
    strokeWidth: 0,
    discrete: []
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  }
}));
</script>

<style scoped>
.chart-shell {
  margin-bottom: 15px;
  display: grid;
  gap: 6px;
}

.chart-panel {
  padding: 0;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.light-mode .chart-panel {
  background: rgba(0, 0, 0, 0.05);
}

.chart-range {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px 0 10px;
  font-size: 11px;
  font-weight: 600;
  opacity: 0.75;
}

.range-label {
  text-transform: uppercase;
  letter-spacing: 0.6px;
}

.range-value {
  font-variant-numeric: tabular-nums;
}

.range-arrow {
  opacity: 0.6;
}

.range-spacer {
  flex: 1;
}

.range-duration {
  font-variant-numeric: tabular-nums;
  opacity: 0.7;
}

.range-dot {
  opacity: 0.5;
}

.range-confidence {
  font-variant-numeric: tabular-nums;
  opacity: 0.7;
}

.chart-empty {
  text-align: center;
  padding: 16px;
  opacity: 0.6;
}

.chart-footer {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  opacity: 0.6;
  padding: 0 6px;
}
</style>
