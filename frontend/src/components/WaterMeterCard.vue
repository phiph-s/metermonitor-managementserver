<template>
  <n-flex vertical style="max-width: 100%;">
    <n-card size="small" class="meter-card" :class="{ 'state-error': hasError, 'state-warning': !hasBB && !setup }" :style="setup?'width: 375px;':''">
      <template #header>
        <div class="card-header">
          <div class="title-group">
            <n-flex :wrap="false">
              <img class="last-image" :src="last_image" :alt="meter_name" v-if="last_image" />
              <div>
                <span class="card-title" :title="meter_name">{{ meter_name }}</span>
                <span class="source-pill" :style="{ '--pill-color': sourceColor, 'margin-right': '4px'}">
                  <n-icon size="14"><component :is="sourceIcon" /></n-icon>
                  <span>{{ sourceLabel }}</span>
                </span>
                <span class="source-pill" :style="{ '--pill-color': '#7e8798' }">
                  <n-icon size="14"><AccessTimeFilled/></n-icon>
                  <n-tooltip trigger="hover" placement="bottom">
                    <template #trigger>
                      <span>{{ last_updated_relative }}</span>
                    </template>
                    {{ last_updated_locale }}
                  </n-tooltip>
                </span>
              </div>
            </n-flex>
          </div>
          <div class="header-meta">

          </div>
        </div>
      </template>

      <template v-if="!setup" #cover>
        <div class="card-type-title" :style="{ '--type-color': meterTypeColor }">
          <span class="type-label">{{ meterTypeLabel }}</span>
          <n-dropdown :options="menuOptions" @select="handleMenuSelect" placement="bottom-end">
            <n-button text class="cover-menu-btn">
              <template #icon>
                <n-icon size="14"><MoreVertFilled /></n-icon>
              </template>
            </n-button>
          </n-dropdown>
        </div>
      </template>

      <SourceCollapse v-if="setup" :source="source"></SourceCollapse>

      <div class="result-row" v-if="last_result != null">
        <div class="stat-chip stat-chip--current">
          <div class="stat-label">Current reading</div>
          <div class="stat-value big">{{ formatConsumption(last_result) }}</div>
        </div>
      </div>

      <div v-if="!setup && stats" class="stats-row">
        <div class="stat-chip">
          <div class="stat-label">Today</div>
          <div class="stat-value">{{ formatConsumption(stats.today_consumption) }}</div>
        </div>
        <div class="stat-chip">
          <div class="stat-label">Daily avg</div>
          <div class="stat-value">{{ formatConsumption(stats.daily_avg) }}</div>
        </div>
        <n-tooltip v-if="stats.extrapolated" trigger="hover" placement="bottom">
          <template #trigger>
            <div class="stat-chip stat-chip--extrapolated">
              <div class="stat-label">Yearly <span class="extrap-mark">~</span></div>
              <div class="stat-value">{{ formatConsumption(stats.yearly_avg) }}</div>
            </div>
          </template>
          Extrapolated from {{ stats.days_of_data }} day{{ stats.days_of_data === 1 ? '' : 's' }} of data
        </n-tooltip>
        <div v-else class="stat-chip">
          <div class="stat-label">Yearly</div>
          <div class="stat-value">{{ formatConsumption(stats.yearly_avg) }}</div>
        </div>
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
import {NCard, NButton, NFlex, NDropdown, NIcon, NTooltip, useDialog} from 'naive-ui';
import {defineProps, computed, ref, onMounted, watch} from 'vue';
import { MoreVertFilled, HomeOutlined, PublicFilled, WifiTetheringOutlined, HelpOutlineOutlined, AccessTimeFilled } from '@vicons/material';
import WifiStatus from "@/components/WifiStatus.vue";
import { useThemeStore } from '@/stores/themeStore';
import { storeToRefs } from 'pinia';
import { getSourceColor, getSourceLabel, normalizeSourceType } from '@/utils/sourceMeta';
import { getMeterUnit, getMeterTypeColor, getMeterTypeLabel } from '@/utils/meterTypeMeta';
import SourceCollapse from "@/components/SourceCollapse.vue";
import { apiService } from '@/services/api';

const themeStore = useThemeStore();
const { } = storeToRefs(themeStore);

const props = defineProps([
    'meter_name',
    'last_updated', // eg "2025-02-04T03:15:31"
    'setup',
    'last_image',
    'last_result',
    'rssi',
    'last_error',
    'has_bbox',
    'source_type',
    'decimals',
    'meter_type',
    'unit'
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

const stats = ref(null);
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

const loadStats = async () => {
  if (props.setup) return;
  try {
    stats.value = await apiService.getJson(`api/watermeters/${props.meter_name}/stats`);
  } catch (e) {
    console.error('Failed to load stats for card:', e);
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
  loadStats();
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

const last_updated_relative = computed(() => {
  if (!props.last_updated) return '';
  const date = new Date(props.last_updated);
  const diffMs = date.getTime() - Date.now();
  const diffSeconds = Math.round(diffMs / 1000);
  const rtf = new Intl.RelativeTimeFormat('en-US', { numeric: 'auto' });
  const absSeconds = Math.abs(diffSeconds);
  if (absSeconds < 20) return 'just now';
  if (absSeconds < 60) return rtf.format(diffSeconds, 'second');
  if (absSeconds < 3600) return rtf.format(Math.round(diffSeconds / 60), 'minute');
  if (absSeconds < 86400) return rtf.format(Math.round(diffSeconds / 3600), 'hour');
  if (absSeconds < 604800) return rtf.format(Math.round(diffSeconds / 86400), 'day');
  if (absSeconds < 2629800) return rtf.format(Math.round(diffSeconds / 604800), 'week');
  return rtf.format(Math.round(diffSeconds / 2629800), 'month');
});

const meterDecimals = computed(() => {
  const raw = Number.isFinite(props.decimals) ? props.decimals : 3;
  return Math.max(0, raw);
});

const meterUnit = computed(() => getMeterUnit(props.meter_type || 'WATER', props.unit));
const meterTypeColor = computed(() => getMeterTypeColor(props.meter_type || 'WATER'));
const meterTypeLabel = computed(() => getMeterTypeLabel(props.meter_type || 'WATER'));

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

const formatConsumption = (rawValue) => {
  if (rawValue == null) return '—';
  return (rawValue / meterScale.value).toFixed(Math.min(meterDecimals.value, 3)) + ' ' + meterUnit.value;
};
</script>

<style scoped>
.meter-card {
  max-width: 100%;
  width: 375px;
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
  align-self: flex-start;
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

.last-image-row {
  margin-top: 6px;
  margin-bottom: 4px;
  border-radius: 8px;
  overflow: hidden;
}

.last-image {
  height: 64px;
  width:64px;
  display: block;
  object-fit: cover;
  border-radius: 8px;
}

.result-row {
  display: flex;
  margin: 4px 0 0 0;
}

.stat-chip--current {
  width: 100%;
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

.stats-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-top: 10px;
}

.stat-chip {
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.06);
}

.light-mode .stat-chip {
  background: rgba(0, 0, 0, 0.04);
}

.stat-chip--extrapolated {
  outline: 1px dashed rgba(255, 126, 0, 0.8);
}

.stat-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  opacity: 0.55;
  font-weight: 600;
  white-space: nowrap;
}

.extrap-mark {
  color: rgb(232, 108, 0);
  font-size: 12px;
}

.stat-value {
  font-size: 12px;
  font-weight: 700;
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stat-value.big{
  font-size: 16px;
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

.cover-menu-btn {
  opacity: 0.7;
  flex-shrink: 0;
  padding-right: 4px;
}

.cover-menu-btn:hover {
  opacity: 1;
}
</style>
