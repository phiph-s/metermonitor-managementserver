<template>
  <div ref="scrollRoot" class="bglight eval-list">
    <div v-if="evaluations.length === 0" style="padding: 20px; width: 100%; margin-top: 20%;">
      <n-empty description="Waiting for the first images...">
      </n-empty>
    </div>
    <n-flex v-else vertical align="start" :size="0">
      <div
        v-for="[i, evaluation] in evaluations.entries()"
        :key="i"
        :class="{ outdated: evaluation.outdated, rejected: !evaluation.total_confidence, item: true }"
        @click="openDetailDialog(evaluation.id)"
        style="cursor: pointer;"
      >
        <div
          v-if="evaluation.outdated && !evaluations[i - 1]?.outdated"
          class="outdated-separator"
        >
          Outdated - The setup configuration has changed since.
          <n-icon size="16" style="margin-left: 4px;">
            <ArrowDownwardOutlined />
          </n-icon>
        </div>
        <n-flex :class="{ redbg: evaluation.result == null, econtainer: true }" vertical :size="0">
          <n-flex align="center">
            <div class="timestamp" :title="formattedTimestampAbsolute(evaluation.timestamp)">
              {{ formattedTimestamp(evaluation.timestamp) }}
            </div>
            <div v-if="evaluation.result" class="corrected-block">
              <div class="result-digits">
                <span
                  v-for="[i, digit] in (evaluation.result + '').padStart(evaluation.th_digits.length, '0').split('').entries()"
                  :key="i + 'f'"
                  :class="{
                    'google-sans-code': true,
                    adjustment: true,
                    red: digit !== evaluation.predictions[i][0][0],
                    blue: evaluation.predictions[i][0][0] === 'r',
                    orange: evaluation.denied_digits[i] && evaluation.predictions[i][0][0] != digit
                  }"
                >
                  <template v-if="isDecimalSeparatorIndex(i, evaluation.th_digits.length)">
                    {{ digit }},
                  </template>
                  <template v-else>
                    {{ digit }}
                  </template>
                </span>
                <span class="adjustment unit">{{ meterUnit }}</span>
              </div>
            </div>

            <template v-if="evaluation.total_confidence">
              <span :style="{ color: getColor(evaluation.total_confidence) }">
                <b>{{ (evaluation.total_confidence * 100).toFixed(1) }}</b>%
              </span>
            </template>
            <div v-else class="rejected-label">
              Rejected
            </div>

            <n-flex class="digit-groups" aria-label="Digits with prediction and confidence">
              <div
                v-for="(base64, j) in evaluation.th_digits_inverted"
                :key="evaluation.id + '-' + j"
                class="digit-group"
              >
                <img
                  class="digit theme-revert"
                  :src="'data:image/png;base64,' + base64"
                  alt="Watermeter"
                />
                <div class="digit-meta">
                  <n-tooltip>
                    <template #trigger>
                      <span class="digit-pred">
                        {{ (evaluation.predictions[j]?.[0]?.[0] === 'r') ? '↕' : evaluation.predictions[j]?.[0]?.[0] }}
                      </span>
                    </template>
                    <span v-if="evaluation.predictions[j]">
                      {{ (evaluation.predictions[j][1][0] === 'r') ? '↕' : evaluation.predictions[j][1][0] }}: {{ (evaluation.predictions[j][1][1] * 100).toFixed(1) }}%<br>
                      {{ (evaluation.predictions[j][2][0] === 'r') ? '↕' : evaluation.predictions[j][2][0] }}: {{ (evaluation.predictions[j][2][1] * 100).toFixed(1) }}%
                    </span>
                  </n-tooltip>
                  <span
                    class="digit-conf"
                    :style="{ color: getColor(evaluation.predictions[j]?.[0]?.[1] || 0), textDecoration: evaluation.denied_digits[j]? 'line-through' : 'none' }"
                  >
                    {{ evaluation.predictions[j] ? Math.round(evaluation.predictions[j][0][1] * 100) : '--' }}
                  </span>
                </div>
              </div>
            </n-flex>

            <n-button
              size="small"
              quaternary
              circle
              @click.stop="openUploadDialog(evaluation.colored_digits, evaluation.th_digits, name, evaluation.predictions)"
            >
              <template #icon>
                <n-icon><ArchiveOutlined /></n-icon>
              </template>
            </n-button>

          </n-flex>
        </n-flex>
      </div>
      <div ref="sentinel" class="scroll-sentinel">
        <span v-if="loadingMore" class="loading-hint">Loading more...</span>
      </div>
    </n-flex>

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
  padding: 0;
  width: fit-content;
  max-width: 100%;
}

.item {
  position: relative;
  width: fit-content;
}

.item.outdated .econtainer {
  border-left: 4px solid rgba(255, 180, 70, 0.9);
  background: rgba(255, 190, 90, 0.08);
}

.item.rejected .econtainer {
  border-left: 4px solid rgba(255, 80, 80, 0.95);
  background: rgba(255, 80, 80, 0.12);
}

.rejected-label {
  color: #ff6b6b;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.outdated-separator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: rgba(255, 180, 80, 0.9);
  background: rgba(255, 190, 90, 0.08);
  border-top: 1px solid rgba(255, 180, 70, 0.4);
  border-bottom: 1px solid rgba(255, 180, 70, 0.4);
}

.light-mode .outdated-separator {
  color: rgba(150, 85, 0, 0.9);
  background: rgba(255, 190, 90, 0.2);
  border-top: 1px solid rgba(180, 110, 30, 0.25);
  border-bottom: 1px solid rgba(180, 110, 30, 0.25);
}

.scroll-sentinel {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding: 14px 0 10px;
  opacity: 0.5;
  min-height: 32px;
}

.loading-hint {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.2em;
}

.list-row {
  display: grid;
  grid-template-columns: 200px 140px auto 48px;
  column-gap: 8px;
  row-gap: 6px;
  width: 100%;
  align-items: center;
  justify-items: start;
  justify-content: start;
}

.cell {
  display: flex;
  gap: 6px;
  width: 100%;
  min-width: 0;
}

.meta-cell {
  gap: 8px;
  align-items: flex-start;
}

.timestamp {
  font-weight: 700;
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  opacity: 0.8;
}

.conf-cell .conf-values {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 18px;
}

.conf-primary {
  font-size: 20px;
}

.conf-secondary {
  font-size: 12px;
  opacity: 0.7;
  cursor: help;
}

.digits-cell {
  justify-content: flex-start;
}

.digit {
  height: 32px;
  mix-blend-mode: screen;
}

.label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  opacity: 0.55;
}

.digit-groups {
}

.digit-group {
  display: grid;
  grid-template-columns: auto;
  grid-template-rows: auto auto;
  justify-items: center;
  align-items: center;
  gap: 2px;
  padding: 0;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  min-width: 48px;
}

.light-mode .digit-group {
  background: rgba(0, 0, 0, 0.04);
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.digit-meta {
  display: flex;
  align-items: baseline;
  gap: 0;
  padding-top: 0px;
  border-top: 1px solid rgba(255, 255, 255, 0.14);
  width: 100%;
  justify-content: center;
}

.light-mode .digit-meta {
  border-top: 1px solid rgba(0, 0, 0, 0.12);
}

.digit-pred {
  font-size: 14px;
  font-weight: 700;
}

.digit-conf {
  font-size: 10px;
  font-weight: 800;
  padding-left: 6px;
  margin-left: 6px;
  border-left: 1px solid rgba(255, 255, 255, 0.2);
}

.light-mode .digit-conf {
  border-left: 1px solid rgba(0, 0, 0, 0.2);
}

.corrected-block .result-digits {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: nowrap;
}

.adjustment {
  font-size: 20px;
  margin: 0;
  color: rgba(255, 255, 255, 0.7);
}

.light-mode .adjustment {
  color: rgba(0, 0, 0, 0.7);
}

.unit {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-left: 6px;
}

.action-cell {
  align-items: flex-start;
  justify-content: flex-start;
}

.red {
  color: #ff6b6b;
}

.blue {
  color: #3aa0ff;
}

.orange {
  color: #ffb45a;
}

@media (max-width: 1400px) {
  .list-row {
    grid-template-columns: 190px 130px minmax(300px, 1fr) 40px;
  }
}

@media (max-width: 1100px) {
  .list-row {
    grid-template-columns: 190px 140px minmax(240px, 1fr) 40px;
    grid-auto-rows: auto;
  }
}

.econtainer {
  width: fit-content;
  padding: 16px 14px;
  margin: 0;
  border-radius: 0;
  background-color: transparent;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  align-items: flex-start;
}

.light-mode .econtainer {
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.redbg {
  background-color: rgba(255, 0, 0, 0.05);
}

</style>
