<template>
  <div ref="scrollRoot" class="eval-list">
    <div v-if="evaluations.length === 0" style="padding: 20px; width: 100%; margin-top: 20%;">
      <n-empty description="Waiting for the first images..." />
    </div>

    <table v-else class="eval-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Reading</th>
          <th>Confidence</th>
          <th>Digits</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <template v-for="[i, evaluation] in evaluations.entries()" :key="i">
          <tr v-if="evaluation.outdated && !evaluations[i - 1]?.outdated" class="outdated-separator-row">
            <td colspan="5">
              <span>Outdated — setup configuration has changed since</span>
              <n-icon size="14" style="margin-left: 6px; vertical-align: middle;">
                <ArrowDownwardOutlined />
              </n-icon>
            </td>
          </tr>
          <tr
            :class="{ outdated: evaluation.outdated, rejected: !evaluation.total_confidence }"
            class="eval-row"
            @click="openDetailDialog(evaluation.id)"
          >
            <td class="td-time">
              <span class="timestamp" :title="formattedTimestampAbsolute(evaluation.timestamp)">
                {{ formattedTimestamp(evaluation.timestamp) }}
              </span>
            </td>
            <td class="td-reading">
              <div v-if="evaluation.result" class="result-digits">
                <span
                  v-for="[j, digit] in (evaluation.result + '').padStart(evaluation.th_digits.length, '0').split('').entries()"
                  :key="j + 'f'"
                  :class="{
                    'google-sans-code': true,
                    adjustment: true,
                    red: digit !== evaluation.predictions[j][0][0],
                    blue: evaluation.predictions[j][0][0] === 'r',
                    orange: evaluation.denied_digits[j] && evaluation.predictions[j][0][0] != digit
                  }"
                >
                  <template v-if="isDecimalSeparatorIndex(j, evaluation.th_digits.length)">{{ digit }},</template>
                  <template v-else>{{ digit }}</template>
                </span>
                <span class="adjustment unit">{{ meterUnit }}</span>
              </div>
              <span v-else class="no-reading">—</span>
            </td>
            <td class="td-conf">
              <span v-if="evaluation.total_confidence" :style="{ color: getColor(evaluation.total_confidence) }">
                <b>{{ (evaluation.total_confidence * 100).toFixed(1) }}</b>%
              </span>
              <span v-else class="rejected-label">Rejected</span>
            </td>
            <td class="td-digits">
              <div class="digit-groups" aria-label="Digits with prediction and confidence">
                <div
                  v-for="(base64, j) in evaluation.th_digits_inverted"
                  :key="evaluation.id + '-' + j"
                  class="digit-group"
                >
                  <img class="digit theme-revert" :src="'data:image/png;base64,' + base64" alt="digit" />
                  <div class="digit-meta">
                    <n-tooltip>
                      <template #trigger>
                        <span class="digit-pred">
                          {{ evaluation.predictions[j]?.[0]?.[0] === 'r' ? '↕' : evaluation.predictions[j]?.[0]?.[0] }}
                        </span>
                      </template>
                      <span v-if="evaluation.predictions[j]">
                        {{ evaluation.predictions[j][1][0] === 'r' ? '↕' : evaluation.predictions[j][1][0] }}: {{ (evaluation.predictions[j][1][1] * 100).toFixed(1) }}%<br>
                        {{ evaluation.predictions[j][2][0] === 'r' ? '↕' : evaluation.predictions[j][2][0] }}: {{ (evaluation.predictions[j][2][1] * 100).toFixed(1) }}%
                      </span>
                    </n-tooltip>
                    <span
                      class="digit-conf"
                      :style="{ color: getColor(evaluation.predictions[j]?.[0]?.[1] || 0), textDecoration: evaluation.denied_digits[j] ? 'line-through' : 'none' }"
                    >
                      {{ evaluation.predictions[j] ? Math.round(evaluation.predictions[j][0][1] * 100) : '--' }}
                    </span>
                  </div>
                </div>
              </div>
            </td>
            <td class="td-action" @click.stop>
              <n-button size="small" quaternary circle
                @click="openUploadDialog(evaluation.colored_digits, evaluation.th_digits, name, evaluation.predictions)"
              >
                <template #icon><n-icon><ArchiveOutlined /></n-icon></template>
              </n-button>
            </td>
          </tr>
        </template>
        <tr v-if="hasMore" ref="sentinel" class="scroll-sentinel-row">
          <td colspan="5">
            <span v-if="loadingMore" class="loading-hint">Loading more...</span>
          </td>
        </tr>
      </tbody>
    </table>

    <EvaluationDetailDialog
      v-model:show="showDetailDialog"
      :evaluation-id="selectedEvaluationId"
      :meter-name="props.name"
      :meter-type="props.meterType"
      :unit="props.unit"
    />
  </div>
</template>

<script setup>
import {defineProps, h, defineEmits, ref, onMounted, onUnmounted, watch, computed} from 'vue';
import { getMeterUnit } from '@/utils/meterTypeMeta';
import {NFlex, NTooltip, NEmpty, NButton, NIcon, useDialog} from 'naive-ui';
import { ArchiveOutlined, ArrowDownwardOutlined } from '@vicons/material';
import DatasetUploader from "@/components/DatasetUploader.vue";
import EvaluationDetailDialog from "@/components/EvaluationDetailDialog.vue";

const dialog = useDialog();
const emit = defineEmits(['loadMore', 'datasetUploaded']);
const showDetailDialog = ref(false);
const selectedEvaluationId = ref(null);
const scrollRoot = ref(null);
const sentinel = ref(null);
const loadingMore = ref(false);
let observer;

const props = defineProps({
  evaluations: {
    type: Array,
    default: () => []
  },
  name: {
    type: String,
    default: ''
  },
  decimals: {
    type: Number,
    default: 3
  },
  meterType: {
    type: String,
    default: 'WATER'
  },
  unit: {
    type: String,
    default: null
  },
  hasMore: {
    type: Boolean,
    default: true
  }
});

const meterUnit = computed(() => getMeterUnit(props.meterType, props.unit));

const getDecimals = (digitLength) => {
  const maxDigits = Number.isFinite(digitLength) ? digitLength : 0;
  const raw = Number.isFinite(props.decimals) ? props.decimals : 3;
  return Math.max(0, Math.min(raw, maxDigits));
};

const isDecimalSeparatorIndex = (idx, digitLength) => {
  const decimals = getDecimals(digitLength);
  if (decimals === 0) return false;
  return idx === digitLength - decimals - 1;
};

const getColor = (value) => {
  // Clamp value between 0 and 1 and map it to a hue (red to green)
  value = Math.max(0, Math.min(1, value));
  const hue = value * 120;
  return `hsl(${hue}, 100%, 40%)`;
};

const openUploadDialog = (colored, thresholded, name, values) => {
  const setvalues = values.map(sub => sub[0][0]);
  let dialogInstance;
  dialogInstance = dialog.info({
    title: 'Upload Dataset',
    content: () => h(DatasetUploader , {
      colored,
      thresholded,
      name,
      setvalues,
      onClose: () => {
        dialogInstance?.destroy();
      },
      onUploaded: () => {
        emit('datasetUploaded');
      }
    }),
    closable: true,
    style: { width: '600px' }
  });
};


const formattedTimestampAbsolute = (ts) => {
  const date = new Date(ts);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' · ' + date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
};

const formattedTimestamp = (ts) => {
  const date = new Date(ts);
  const diffMs = date.getTime() - Date.now();
  const diffSeconds = Math.round(diffMs / 1000);
  const rtf = new Intl.RelativeTimeFormat("en-US", { numeric: 'auto' });
  const absSeconds = Math.abs(diffSeconds);

  if (absSeconds < 20) return 'just now';
  if (absSeconds < 60) return rtf.format(diffSeconds, 'second');
  if (absSeconds < 3600) return rtf.format(Math.round(diffSeconds / 60), 'minute');
  if (absSeconds < 86400) return rtf.format(Math.round(diffSeconds / 3600), 'hour');
  if (absSeconds < 604800) return rtf.format(Math.round(diffSeconds / 86400), 'day');
  if (absSeconds < 2629800) return rtf.format(Math.round(diffSeconds / 604800), 'week');
  return rtf.format(Math.round(diffSeconds / 2629800), 'month');
};

const openDetailDialog = (evalId) => {
  selectedEvaluationId.value = evalId;
  showDetailDialog.value = true;
};

const triggerLoadMore = () => {
  if (loadingMore.value) return;
  loadingMore.value = true;
  emit('loadMore');
};

onMounted(() => {
  observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      triggerLoadMore();
    }
  }, {
    root: scrollRoot.value,
    rootMargin: '200px',
    threshold: 0.1
  });
  if (sentinel.value) {
    observer.observe(sentinel.value);
  }
});

watch(
  () => sentinel.value,
  (next, prev) => {
    if (!observer) return;
    if (prev) observer.unobserve(prev);
    if (next) observer.observe(next);
  }
);

onUnmounted(() => {
  if (observer) observer.disconnect();
});

watch(
  () => props.evaluations.length,
  () => {
    loadingMore.value = false;
  }
);
</script>

<style scoped>
.eval-list {
  height: 100%;
  overflow-y: auto;
  overflow-x: auto;
}

/* ── Table layout ── */
.eval-table {
  width: 100%;
  border-collapse: collapse;
  table-layout: auto;
}

thead tr {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--n-color, #1a1a1a);
}

.light-mode thead tr {
  background: #f5f5f5;
}

thead th {
  padding: 8px 14px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  opacity: 0.5;
  text-align: left;
  white-space: nowrap;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.light-mode thead th {
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

/* ── Body rows ── */
.eval-row {
  cursor: pointer;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  transition: background 0.1s;
}

.light-mode .eval-row {
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.eval-row:hover {
  background: rgba(255, 255, 255, 0.04);
}

.light-mode .eval-row:hover {
  background: rgba(0, 0, 0, 0.03);
}

.eval-row.outdated {
  border-left: 3px solid rgba(255, 180, 70, 0.8);
  background: rgba(255, 190, 90, 0.05);
}

.eval-row.rejected {
  border-left: 3px solid rgba(255, 80, 80, 0.9);
  background: rgba(255, 80, 80, 0.07);
}

.eval-row td {
  padding: 10px 14px;
  vertical-align: middle;
}

/* ── Outdated separator row ── */
.outdated-separator-row td {
  padding: 7px 14px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: rgba(255, 180, 80, 0.9);
  background: rgba(255, 190, 90, 0.08);
  border-top: 1px solid rgba(255, 180, 70, 0.3);
  border-bottom: 1px solid rgba(255, 180, 70, 0.3);
}

.light-mode .outdated-separator-row td {
  color: rgba(150, 85, 0, 0.9);
  background: rgba(255, 190, 90, 0.18);
  border-color: rgba(180, 110, 30, 0.25);
}

/* ── Sentinel / loading ── */
.scroll-sentinel-row td {
  padding: 14px;
  text-align: center;
}

.loading-hint {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  opacity: 0.5;
}

/* ── Cell content ── */
.td-time {
  white-space: nowrap;
}

.timestamp {
  font-weight: 700;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  opacity: 0.8;
}

.td-reading {
  white-space: nowrap;
}

.result-digits {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: nowrap;
}

.no-reading {
  opacity: 0.4;
}

.adjustment {
  font-size: 20px;
  color: rgba(255, 255, 255, 0.7);
}

.light-mode .adjustment {
  color: rgba(0, 0, 0, 0.7);
}

.unit {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-left: 4px;
}

.td-conf {
  white-space: nowrap;
}

.rejected-label {
  color: #ff6b6b;
  font-size: 13px;
  font-weight: 700;
}

.td-digits {
  padding-top: 8px;
  padding-bottom: 8px;
}

.digit-groups {
  display: flex;
  gap: 4px;
  flex-wrap: nowrap;
}

.digit-group {
  display: grid;
  grid-template-rows: auto auto;
  justify-items: center;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  min-width: 44px;
  overflow: hidden;
}

.light-mode .digit-group {
  background: rgba(0, 0, 0, 0.04);
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.digit {
  height: 32px;
  mix-blend-mode: screen;
}

.digit-meta {
  display: flex;
  align-items: baseline;
  width: 100%;
  justify-content: center;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
}

.light-mode .digit-meta {
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.digit-pred {
  font-size: 13px;
  font-weight: 700;
}

.digit-conf {
  font-size: 10px;
  font-weight: 800;
  padding-left: 5px;
  margin-left: 5px;
  border-left: 1px solid rgba(255, 255, 255, 0.18);
}

.light-mode .digit-conf {
  border-left: 1px solid rgba(0, 0, 0, 0.18);
}

.td-action {
  width: 40px;
  text-align: center;
}

.red   { color: #ff6b6b; }
.blue  { color: #3aa0ff; }
.orange { color: #ffb45a; }
</style>
