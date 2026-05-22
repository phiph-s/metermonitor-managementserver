<template>
  <n-modal v-model:show="show" preset="card" title="Evaluation Details" style="max-width: 900px;" :mask-closable="true">
    <n-spin :show="loading">
      <div v-if="evaluation" class="eval-detail">
        <!-- Header with timestamp and confidence -->
        <n-space justify="space-between" align="center" style="margin-bottom: 20px;">
          <div>
            <h3 style="margin: 0;">{{ formattedTimestamp(evaluation.timestamp) }}</h3>
            <n-tag v-if="evaluation.outdated" type="warning" size="small" style="margin-top: 4px;">
              OUTDATED
            </n-tag>
          </div>
          <div v-if="evaluation.total_confidence" class="conf-display">
            <div style="font-size: 12px; opacity: 0.7;">Confidence</div>
            <div :style="{ color: getColor(evaluation.total_confidence), fontSize: '24px', fontWeight: 'bold' }">
              {{ (evaluation.total_confidence * 100).toFixed(1) }}%
            </div>
          </div>
          <div v-else class="rejected-badge">
            REJECTED
          </div>
        </n-space>

        <!-- Digit details -->
        <n-card size="small" title="Digit Analysis" embedded style="margin-bottom: 16px;">
          <n-space vertical>
            <!-- Digit thumbnails -->
            <div>
              <div style="font-size: 12px; opacity: 0.7; margin-bottom: 8px;">Segmented Digits</div>
              <n-flex class="theme-revert">
                <img
                  v-for="(base64, i) in evaluation.th_digits_inverted"
                  :key="'digit-' + i"
                  class="digit-thumb"
                  :src="'data:image/png;base64,' + base64"
                  alt="Digit"
                />
              </n-flex>
            </div>

            <!-- Predictions table -->
            <n-table :bordered="false" :single-line="false" size="small">
              <thead>
                <tr>
                  <th>Position</th>
                  <th>Top Prediction</th>
                  <th>Confidence</th>
                  <th>2nd Choice</th>
                  <th>3rd Choice</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(digit, i) in evaluation.predictions" :key="'pred-' + i">
                  <td><b>{{ i + 1 }}</b></td>
                  <td>
                    <span class="prediction-value">
                      {{ (digit[0][0] === 'r') ? '↕ (rotating)' : digit[0][0] }}
                    </span>
                  </td>
                  <td>
                    <span :style="{ color: getColor(digit[0][1]), fontWeight: 'bold' }">
                      {{ (digit[0][1] * 100).toFixed(1) }}%
                    </span>
                  </td>
                  <td style="opacity: 0.6;">
                    {{ (digit[1][0] === 'r') ? '↕' : digit[1][0] }}: {{ (digit[1][1] * 100).toFixed(1) }}%
                  </td>
                  <td style="opacity: 0.6;">
                    {{ (digit[2][0] === 'r') ? '↕' : digit[2][0] }}: {{ (digit[2][1] * 100).toFixed(1) }}%
                  </td>
                  <td>
                    <n-tag v-if="evaluation.denied_digits && evaluation.denied_digits[i]" type="warning" size="small">
                      Denied
                    </n-tag>
                    <n-tag v-else type="success" size="small">
                      Accepted
                    </n-tag>
                  </td>
                </tr>
              </tbody>
            </n-table>
          </n-space>
        </n-card>

        <!-- Corrected Result -->
        <n-card v-if="evaluation.result" size="small" title="Corrected Result" embedded>
          <n-space align="center" size="large">
            <span
              v-for="(digit, i) in (evaluation.result + '').padStart(evaluation.th_digits.length, '0').split('')"
              :key="'result-' + i"
              :class="{
                'result-digit': true,
                'red': digit !== evaluation.predictions[i][0][0],
                'blue': evaluation.predictions[i][0][0] === 'r',
                'orange': evaluation.denied_digits[i] && evaluation.predictions[i][0][0] != digit
              }"
            >
              <template v-if="i === evaluation.th_digits.length - 4">
                {{ digit }}<span style="opacity: 0.5;">,</span>
              </template>
              <template v-else>
                {{ digit }}
              </template>
            </span>
            <span style="font-size: 24px; opacity: 0.7;">{{ meterUnit }}</span>
          </n-space>
        </n-card>

        <n-card size="small" title="Correction Metadata" embedded style="margin-top: 16px;">
          <n-grid cols="2 s:1 m:2 l:2" x-gap="16" y-gap="8">
            <n-grid-item>
              <div class="meta-label">Flow rate</div>
              <div class="meta-value">{{ formatFlowRate(evaluation.flow_rate_m3h) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Delta</div>
              <div class="meta-value">{{ formatDelta(evaluation.delta_m3) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Delta (raw)</div>
              <div class="meta-value">{{ formatInt(evaluation.delta_raw) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Time diff</div>
              <div class="meta-value">{{ formatMinutes(evaluation.time_diff_min) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Rejection reason</div>
              <div class="meta-value">{{ formatRejectionReason(evaluation.rejection_reason) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Negative correction</div>
              <div class="meta-value">{{ formatBool(evaluation.negative_correction_applied) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Fallback digits</div>
              <div class="meta-value">{{ formatInt(evaluation.fallback_digit_count) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Digits changed vs last</div>
              <div class="meta-value">{{ formatInt(evaluation.digits_changed_vs_last) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Digits changed vs top pred</div>
              <div class="meta-value">{{ formatInt(evaluation.digits_changed_vs_top_pred) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Prediction ranks used</div>
              <div class="meta-value">{{ formatRankCounts(evaluation.prediction_rank_used_counts) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Denied digits count</div>
              <div class="meta-value">{{ formatInt(evaluation.denied_digits_count) }}</div>
            </n-grid-item>
            <n-grid-item>
              <div class="meta-label">Timestamp adjusted</div>
              <div class="meta-value">{{ formatBool(evaluation.timestamp_adjusted) }}</div>
            </n-grid-item>
          </n-grid>
        </n-card>
      </div>
    </n-spin>
  </n-modal>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import { getMeterUnit } from '@/utils/meterTypeMeta';
import {
  NModal,
  NSpace,
  NCard,
  NGrid,
  NGridItem,
  NEmpty,
  NFlex,
  NTable,
  NTag,
  NSpin,
  NTooltip
} from 'naive-ui';

const props = defineProps({
  show: { type: Boolean, default: false },
  evaluationId: { type: Number, default: null },
  meterName: { type: String, required: true },
  meterType: { type: String, default: 'WATER' },
  unit: { type: String, default: null }
});

const emit = defineEmits(['update:show']);

const show = computed({
  get: () => props.show,
  set: (v) => emit('update:show', v)
});

const loading = ref(false);
const evaluation = ref(null);
const host = import.meta.env.VITE_HOST;

const getColor = (value) => {
  value = Math.max(0, Math.min(1, value));
  const hue = value * 120;
  return `hsl(${hue}, 100%, 40%)`;
};

const formattedTimestamp = (ts) => {
  const date = new Date(ts);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' · ' + date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined) return '—';
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return '—';
  return numeric.toFixed(digits);
};

const formatInt = (value) => {
  if (value === null || value === undefined) return '—';
  return value;
};

const formatBool = (value) => {
  if (value === null || value === undefined) return '—';
  return value ? 'Yes' : 'No';
};

const formatFlowRate = (value) => {
  if (value === null || value === undefined) return '—';
  return `${formatNumber(value, 3)} ${meterUnit.value}/h`;
};

const meterUnit = computed(() => getMeterUnit(props.meterType, props.unit));

const formatDelta = (value) => {
  if (value === null || value === undefined) return '—';
  return `${formatNumber(value, 3)} ${meterUnit.value}`;
};

const formatMinutes = (value) => {
  if (value === null || value === undefined) return '—';
  return `${formatNumber(value, 2)} min`;
};

const formatRankCounts = (counts) => {
  if (!Array.isArray(counts)) return '—';
  const [rank1 = 0, rank2 = 0, rank3 = 0] = counts;
  return `#1 ${rank1}, #2 ${rank2}, #3 ${rank3}`;
};

const formatRejectionReason = (reason) => {
  if (!reason) return '—';
  const labels = {
    no_history: 'No history',
    time_diff_zero: 'Time diff = 0',
    flow_rate_high: 'Flow rate too high',
    negative_flow: 'Negative flow',
    fallback_digit: 'Fallback digit'
  };
  return labels[reason] || reason;
};

const loadEvaluation = async () => {
  if (!props.evaluationId || !props.meterName) return;

  loading.value = true;
  try {
    const response = await fetch(
      `${host}api/watermeters/${props.meterName}/evals/${props.evaluationId}`,
      {
        headers: { 'secret': localStorage.getItem('secret') }
      }
    );
    if (response.ok) {
      evaluation.value = await response.json();
    }
  } catch (e) {
    console.error('Failed to load evaluation details:', e);
  } finally {
    loading.value = false;
  }
};

watch(() => props.show, (newVal) => {
  if (newVal) {
    loadEvaluation();
  }
});
</script>

<style scoped>
.eval-detail {
  min-height: 200px;
}

.conf-display {
  text-align: right;
  padding: 12px;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
}

.rejected-badge {
  color: #d03050;
  font-size: 24px;
  font-weight: bold;
  padding: 12px;
  background-color: rgba(208, 48, 80, 0.1);
  border-radius: 8px;
}

.digit-thumb {
  height: 50px;
  margin: 4px;
  mix-blend-mode: screen;
  opacity: 0.8;
  border-radius: 4px;
}

.prediction-value {
  font-size: 18px;
  font-weight: bold;
  font-family: 'Google Sans', monospace;
}

.result-digit {
  font-size: 32px;
  font-weight: bold;
  font-family: 'Google Sans', monospace;
  color: rgba(255, 255, 255, 0.9);
}

.light-mode .result-digit {
  color: rgba(0, 0, 0, 0.9);
}

.result-digit.red {
  color: #d03050;
}

.result-digit.blue {
  color: dodgerblue;
}

.result-digit.orange {
  color: orange;
}

.meta-label {
  font-size: 12px;
  opacity: 0.7;
}

.meta-value {
  font-size: 14px;
  font-weight: 600;
}
</style>
