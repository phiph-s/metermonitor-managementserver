<template>
  <div class="chart-shell">
    <!-- Tab toggle -->
    <div class="chart-tabs">
      <button
        class="chart-tab"
        :class="{ active: activeTab === 'detailed' }"
        @click="activeTab = 'detailed'"
      >Detailed</button>
      <button
        class="chart-tab"
        :class="{ active: activeTab === 'daily' }"
        @click="activeTab = 'daily'"
      >Daily</button>
    </div>

    <!-- Detailed chart -->
    <template v-if="activeTab === 'detailed'">
      <div class="chart-panel">
        <div class="chart-range">
          <span class="range-label">Usage</span>

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
          height="160"
          type="line"
          :options="combinedChartOptions"
          :series="combinedSeries"
          :key="'detailed-' + isDark"
        />
        <div v-else class="chart-empty">No history data available</div>
      </div>
      <div v-if="combinedSeries.length > 0" class="chart-footer">
        <span>{{ detailedLeft }}</span>
        <span>{{ detailedMid }}</span>
        <span>{{ detailedRight }}</span>
      </div>
    </template>

    <!-- Daily chart -->
    <template v-else>
      <div class="chart-panel">
        <div class="chart-range">
          <span class="range-label">Daily</span>
          <span class="range-spacer"></span>
          <span class="range-duration">{{ dailyDurationLabel }}</span>
          <span class="range-dot">·</span>
          <span class="range-confidence">avg {{ formatUsage(dailyAvgConsumption) }}</span>
        </div>
        <apexchart
          v-if="dailySeries.length > 0"
          width="100%"
          height="160"
          type="area"
          :options="dailyChartOptions"
          :series="dailySeries"
          :key="'daily-' + isDark"
        />
        <div v-else class="chart-empty">No daily history data available</div>
      </div>
      <div v-if="dailySeries.length > 0" class="chart-footer">
        <span>{{ dailyLeft }}</span>
        <span>{{ dailyMid }}</span>
        <span>{{ dailyRight }}</span>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useThemeStore } from '@/stores/themeStore';
import { storeToRefs } from 'pinia';

const themeStore = useThemeStore();
const { isDark } = storeToRefs(themeStore);

const props = defineProps({
  history: { type: Object, default: null },
  dailyHistory: { type: Object, default: null },
  unit: { type: String, default: 'm³' },
  dailyAvg: { type: Number, default: null },
});

const activeTab = ref('detailed');

// ── Detailed (existing logic) ──────────────────────────────────────────────

const sortedHistory = computed(() => {
  if (!props.history?.history) return [];
  return [...props.history.history].sort((a, b) => new Date(a[1]) - new Date(b[1]));
});

const usageSeries = computed(() => {
  if (!sortedHistory.value.length) return [];
  return [{ name: 'Usage', data: sortedHistory.value.map(item => ({ x: new Date(item[1]).getTime(), y: item[0] / 1000 })) }];
});

const confidenceSeries = computed(() => {
  if (!sortedHistory.value.length) return [];
  return [{ name: 'Confidence', data: sortedHistory.value.map(item => ({ x: new Date(item[1]).getTime(), y: item[2] * 100 })) }];
});

const combinedSeries = computed(() => {
  if (!usageSeries.value.length) return [];
  return [
    { ...usageSeries.value[0], type: 'area' },
    { ...confidenceSeries.value[0], type: 'column' },
  ];
});

const usageMin = computed(() => usageSeries.value.length ? Math.min(...usageSeries.value[0].data.map(p => p.y)) : null);
const usageMax = computed(() => usageSeries.value.length ? Math.max(...usageSeries.value[0].data.map(p => p.y)) : null);
const usageFrom = computed(() => usageSeries.value[0]?.data[0]?.y ?? '—');
const usageTo   = computed(() => { const d = usageSeries.value[0]?.data; return d?.[d.length - 1]?.y ?? '—'; });

const durationLabel = computed(() => {
  if (sortedHistory.value.length < 2) return '—';
  const diffMs = Math.max(0, new Date(sortedHistory.value.at(-1)[1]) - new Date(sortedHistory.value[0][1]));
  const minutes = Math.floor(diffMs / 60000);
  const days = Math.floor(minutes / 1440), hours = Math.floor((minutes % 1440) / 60), mins = minutes % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
});

const confidenceValues = computed(() =>
  sortedHistory.value.map(i => i[2]).filter(v => v != null).map(v => Number(v) * 100).filter(v => !isNaN(v))
);
const averageConfidence = computed(() => {
  if (!confidenceValues.value.length) return '—';
  return `${(confidenceValues.value.reduce((a, b) => a + b, 0) / confidenceValues.value.length).toFixed(1)}%`;
});
const medianConfidence = computed(() => {
  if (!confidenceValues.value.length) return '—';
  const s = [...confidenceValues.value].sort((a, b) => a - b), m = Math.floor(s.length / 2);
  return `${(s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2).toFixed(1)}%`;
});

const detailedRangeMs = computed(() => {
  if (sortedHistory.value.length < 2) return 0;
  return new Date(sortedHistory.value.at(-1)[1]) - new Date(sortedHistory.value[0][1]);
});

const formatDetailedLabel = (v) => {
  if (!v) return '—';
  const date = new Date(v);
  const useTime = detailedRangeMs.value <= 2 * 86400 * 1000;
  if (useTime) {
    const day = date.toLocaleDateString(undefined, { day: '2-digit', month: 'short' });
    const time = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    // Only show date prefix on left/right if range spans multiple days
    return detailedRangeMs.value > 86400 * 1000 ? `${day} ${time}` : time;
  }
  return date.toLocaleDateString(undefined, { day: '2-digit', month: 'short' });
};

const detailedLeft  = computed(() => formatDetailedLabel(sortedHistory.value[0]?.[1]));
const detailedRight = computed(() => formatDetailedLabel(sortedHistory.value.at(-1)?.[1]));
const detailedMid   = computed(() => formatDetailedLabel(sortedHistory.value[Math.floor(sortedHistory.value.length / 2)]?.[1]));

const usageColor      = computed(() => isDark.value ? '#22d3ee' : '#0ea5e9');
const confidenceColor = computed(() => isDark.value ? '#f59e0b' : '#d97706');

const combinedChartOptions = computed(() => ({
  theme: { mode: isDark.value ? 'dark' : 'light' },
  chart: { type: 'line', zoom: { enabled: false }, background: '#00000000', toolbar: { show: false }, animations: { enabled: false }, sparkline: { enabled: true } },
  yaxis: [
    { min: usageMin.value, max: usageMax.value, labels: { show: false } },
    { opposite: true, show: false, min: 0, max: 100 },
  ],
  grid: { show: false, padding: { left: 0, right: 0, top: 4, bottom: 0 } },
  plotOptions: { bar: { columnWidth: '75%', borderRadius: 4 } },
  stroke: { curve: 'smooth', width: [2.5, 0] },
  fill: { type: ['gradient', 'solid'], opacity: [0.18, 0.55], gradient: { shadeIntensity: 0.5, opacityFrom: 0.2, opacityTo: 0.02, stops: [0, 70, 100] } },
  tooltip: { x: { format: 'dd MMM HH:mm' }, y: [{ formatter: v => `${v.toFixed(3)} ${props.unit}` }, { formatter: v => `${v.toFixed(1)}%` }], marker: { show: false } },
  colors: [usageColor.value, confidenceColor.value],
  markers: { size: 0, strokeWidth: 0 },
  dataLabels: { enabled: false },
  legend: { show: false },
}));

// ── Daily ──────────────────────────────────────────────────────────────────

const sortedDaily = computed(() => {
  if (!props.dailyHistory?.history) return [];
  return [...props.dailyHistory.history].sort((a, b) => a[1] < b[1] ? -1 : 1);
});

const dailySeries = computed(() => {
  if (!sortedDaily.value.length) return [];
  return [{ name: 'Reading', data: sortedDaily.value.map(item => ({ x: new Date(item[1]).getTime(), y: item[0] / 1000 })) }];
});

const dailyMin = computed(() => dailySeries.value.length ? Math.min(...dailySeries.value[0].data.map(p => p.y)) : null);
const dailyMax = computed(() => dailySeries.value.length ? Math.max(...dailySeries.value[0].data.map(p => p.y)) : null);
const dailyFrom = computed(() => dailySeries.value[0]?.data[0]?.y ?? '—');
const dailyTo   = computed(() => { const d = dailySeries.value[0]?.data; return d?.[d.length - 1]?.y ?? '—'; });

const dailyAvgConsumption = computed(() => {
  if (props.dailyAvg == null) return null;
  // Backend returns raw integer value; divide by 1000 to match formatUsage expectations
  return props.dailyAvg / 1000;
});

const dailyDurationLabel = computed(() => {
  if (sortedDaily.value.length < 2) return '—';
  const days = Math.round((new Date(sortedDaily.value.at(-1)[1]) - new Date(sortedDaily.value[0][1])) / 86400000);
  return `${days}d`;
});

const dailyLeft  = computed(() => sortedDaily.value[0]?.[1]?.slice(0, 10) ?? '—');
const dailyRight = computed(() => sortedDaily.value.at(-1)?.[1]?.slice(0, 10) ?? '—');
const dailyMid   = computed(() => sortedDaily.value[Math.floor(sortedDaily.value.length / 2)]?.[1]?.slice(0, 10) ?? '—');

const dailyColor = computed(() => isDark.value ? '#a78bfa' : '#7c3aed');

const dailyChartOptions = computed(() => ({
  theme: { mode: isDark.value ? 'dark' : 'light' },
  chart: { type: 'area', zoom: { enabled: false }, background: '#00000000', toolbar: { show: false }, animations: { enabled: false }, sparkline: { enabled: true } },
  yaxis: { min: dailyMin.value, max: dailyMax.value, labels: { formatter: () => '' } },
  stroke: { curve: 'smooth', width: 2 },
  fill: { type: 'gradient', gradient: { shadeIntensity: 0.5, opacityFrom: 0.25, opacityTo: 0.02, stops: [0, 80, 100] } },
  tooltip: { x: { format: 'dd MMM yyyy' }, y: { formatter: v => `${v.toFixed(3)} ${props.unit}` }, marker: { show: false } },
  colors: [dailyColor.value],
  markers: { size: 0 },
  dataLabels: { enabled: false },
  legend: { show: false },
  xaxis: { type: 'datetime' },
}));

const formatUsage = (value) => {
  if (value === null || value === undefined || value === '—') return '—';
  const n = Number(value);
  return isNaN(n) ? '—' : `${n.toFixed(3)} ${props.unit}`;
};
</script>

<style scoped>
.chart-shell {
  margin-bottom: 15px;
  display: grid;
  gap: 6px;
}

.chart-tabs {
  display: flex;
  gap: 2px;
  padding: 0 2px;
}

.chart-tab {
  flex: 1;
  padding: 4px 0;
  border: none;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  cursor: pointer;
  color: inherit;
  opacity: 0.5;
  transition: background 0.15s, opacity 0.15s;
}

.light-mode .chart-tab {
  background: rgba(0, 0, 0, 0.04);
}

.chart-tab:hover {
  opacity: 0.8;
}

.chart-tab.active {
  background: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
  opacity: 1;
}

.chart-panel {
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
  flex-wrap: nowrap;
  gap: 4px;
  padding: 6px 10px 0 10px;
  font-size: 11px;
  font-weight: 600;
  opacity: 0.75;
  overflow: hidden;
}

.range-label {
  text-transform: uppercase;
  letter-spacing: 0.6px;
  white-space: nowrap;
}

.range-value { font-variant-numeric: tabular-nums; white-space: nowrap; }
.range-arrow { opacity: 0.6; flex-shrink: 0; }
.range-spacer { flex: 1; min-width: 4px; }
.range-duration { font-variant-numeric: tabular-nums; opacity: 0.7; white-space: nowrap; }
.range-dot { opacity: 0.5; flex-shrink: 0; }
.range-confidence { font-variant-numeric: tabular-nums; opacity: 0.7; white-space: nowrap; }

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
