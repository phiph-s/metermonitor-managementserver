<template>
  <n-card size="small">
    <template #cover>
      <div class="card-title">
        <span style="font-weight: bolder;">Digit recognition</span><span style="opacity: 0.7"></span>
      </div>
    </template>
    <!-- Search controls -->
    <n-flex align="center" justify="space-between" style="margin-bottom: 8px; padding-top: 16px;">
      <n-flex align="center" :size="8">
        <n-input-group>
          <n-input-group-label size="small">Depth:</n-input-group-label>
          <n-input-number
            v-model:value="searchSteps"
            :min="3"
            :max="25"
            size="small"
            style="width: 80px;"
            :disabled="isDisabled"
          />
          <n-button
            @click="startThresholdSearch"
            size="small"
            :loading="searchingThresholds"
            :disabled="isDisabled"
          >
            Search Thresholds
          </n-button>
        </n-input-group>
      </n-flex>
    </n-flex>

    <!-- Search status indicator -->
    <div v-if="searchingThresholds" style="text-align: center; padding: 8px; margin-bottom: 10px; background: rgba(24, 160, 88, 0.15); border-radius: 4px; font-size: 12px;">
      Searching for optimal thresholds...
    </div>

    <!-- Search result indicator -->
    <div v-if="thresholdSearchResult && !thresholdSearchResult.error && !searchingThresholds"
         style="text-align: center; padding: 8px; margin-bottom: 10px; background: rgba(24, 160, 88, 0.15); border-radius: 4px; font-size: 12px;">
      ✓ Found: Confidence {{ (thresholdSearchResult.avg_confidence * 100).toFixed(1) }}%
    </div>

    <div class="threshold-scroll">
    <n-flex :size="[0,0]" justify="space-around" align="center" style="flex-wrap: nowrap; min-width: min-content;">
      <div>
        <n-flex justify="space-around" size="small" v-if="evaluation">
          <img :style="{ width: digitWidth }" class="digit" v-for="[i,base64] in leadingDigits.entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" />
        </n-flex>
        <br>
        <n-flex justify="space-around" size="small" v-if="tresholdedImages" class="theme-revert">
          <img :style="{ width: digitWidth }" class="digit th" v-for="[i,base64] in leadingThresholded.entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter" />
        </n-flex>
        <br>
        <n-slider :value="currentThreshold" @update:value="updateThreshold" range :step="1" :max="255" @mouseup="sendUpdate" style="" :disabled="isDisabled"/>
        {{currentThreshold[0]}} - {{currentThreshold[1]}}

        <br><br>
        <div class="digit-models-title">Display types</div>
        <n-flex justify="space-around" size="small" v-if="evaluation" class="digit-models-row">
          <n-tooltip v-for="(model, i) in leadingDigitModels" :key="`lead-model-${i}`" trigger="hover">
            <template #trigger>
              <n-button
                size="tiny"
                quaternary
                :class="['digit-model-button', model]"
                :disabled="isDisabled"
                @click="() => toggleDigitModel(i)"
              >
                <template #icon>
                  <n-icon v-if="model !== 'segment'" :component="SwapVertOutlined" />
                  <img v-else :src="sevenSegIcon" alt="7-seg" class="digit-model-icon" />
                </template>
              </n-button>
            </template>
            {{ model === 'segment' ? '7-segment model' : 'Rotating model' }}
          </n-tooltip>
        </n-flex>
      </div>
      <div v-if="showDecimalControl" class="decimal-divider">
        <n-button
          size="tiny"
          quaternary
          :disabled="isDisabled || !canIncreaseDecimals"
          @click="() => changeDecimals(1)"
        >
          <template #icon>
            <n-icon :component="ChevronLeftOutlined" />
          </template>
        </n-button>
        <div class="decimal-dot">.</div>
        <n-button
          size="tiny"
          quaternary
          :disabled="isDisabled || !canDecreaseDecimals"
          @click="() => changeDecimals(-1)"
        >
          <template #icon>
            <n-icon :component="ChevronRightOutlined" />
          </template>
        </n-button>
      </div>
      <div v-if="showLastThreshold">
        <n-flex justify="space-around" size="small" v-if="evaluation">
          <img :style="{ width: digitWidth }" class="digit" v-for="[i,base64] in trailingDigits.entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" />
        </n-flex>
        <br>
        <n-flex justify="space-around" size="small" v-if="tresholdedImages" class="theme-revert">
          <img :style="{ width: digitWidth }" class="digit th" v-for="[i,base64] in trailingThresholded.entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter" />
        </n-flex>
        <br>
        <n-slider :value="currentThresholdLast" @update:value="updateThresholdLast" range :step="1" :max="255" @mouseup="sendUpdate" style="max-width: 150px;" :disabled="isDisabled"/>
        {{currentThresholdLast[0]}} - {{currentThresholdLast[1]}}

        <br><br>
        <div class="digit-models-title">&nbsp;</div>
        <n-flex justify="space-around" size="small" v-if="evaluation" class="digit-models-row">
          <n-tooltip v-for="(model, i) in trailingDigitModels" :key="`trail-model-${i}`" trigger="hover">
            <template #trigger>
              <n-button
                size="tiny"
                quaternary
                :class="['digit-model-button', model]"
                :disabled="isDisabled"
                @click="() => toggleDigitModel(trailingOffset + i)"
              >
                <template #icon>
                  <n-icon v-if="model !== 'segment'" :component="SwapVertOutlined" />
                  <img v-else :src="sevenSegIcon" alt="7-seg" class="digit-model-icon" />
                </template>
              </n-button>
            </template>
            {{ model === 'segment' ? '7-segment model' : 'Rotating model' }}
          </n-tooltip>
        </n-flex>
      </div>
    </n-flex>
    </div>

    <n-divider></n-divider>
    Extraction padding
      <n-slider :value="currentIslandingPadding" @update:value="updateIslandingPadding" :step="1" :max="100" @mouseup="sendUpdate" style="max-width: 150px;" :disabled="isDisabled"/>

    <template #action>
      <n-flex justify="end" size="large">
        <n-button
            @click="() => {emits('reevaluate');emits('next')}"
            round
            :disabled="isDisabled"
        >Apply</n-button>
      </n-flex>
    </template>
  </n-card>
</template>

<script setup>
import {NFlex, NCard, NDivider, NButton, NSlider, NInputNumber, NSpin, NInputGroup, NInputGroupLabel, NIcon, NTooltip} from 'naive-ui';
import {defineProps, defineEmits, ref, watch, onMounted, computed} from 'vue';
import { SwapVertOutlined, ChevronLeftOutlined, ChevronRightOutlined } from '@vicons/material';
import sevenSegIcon from '@/assets/icons/seven-seg.svg';

const props = defineProps([
    'evaluation',
    'threshold',
    'threshold_last',
    'islanding_padding',
    'segments',
    'digitModels',
    'decimals',
    'loading',
    'searchingThresholds',
    'thresholdSearchResult'
]);

const emits = defineEmits(['update', 'reevaluate', 'next', 'searchThresholds', 'update-digit-models', 'update-decimals']);

const currentThreshold = ref(props.threshold);
const currentThresholdLast = ref(props.threshold_last);
const currentIslandingPadding = ref(props.islanding_padding);
const currentDigitModels = ref([]);
const currentDecimals = ref(Number.isFinite(props.decimals) ? props.decimals : 3);

const tresholdedImages = ref([]);
const refreshing = ref(false);
const searchSteps = ref(10);

const isDisabled = computed(() => props.loading || props.searchingThresholds);
const segmentCount = computed(() => {
  const evaluationCount = props.evaluation?.colored_digits?.length || 0;
  const value = props.segments || evaluationCount;
  return value || evaluationCount || 0;
});
const digitCount = computed(() => {
  const evaluationCount = props.evaluation?.colored_digits?.length || 0;
  return evaluationCount || segmentCount.value || 0;
});
const lastDigitCount = computed(() => {
  const decimals = Number.isFinite(currentDecimals.value) ? currentDecimals.value : 0;
  return Math.max(0, Math.min(decimals, digitCount.value));
});
const showLastThreshold = computed(() => lastDigitCount.value > 0);
const showDecimalControl = computed(() => digitCount.value > 0);
const canDecreaseDecimals = computed(() => currentDecimals.value > 0);
const canIncreaseDecimals = computed(() => currentDecimals.value < digitCount.value);
const leadingDigits = computed(() => {
  const digits = props.evaluation?.colored_digits || [];
  return showLastThreshold.value ? digits.slice(0, -lastDigitCount.value) : digits;
});
const trailingDigits = computed(() => {
  const digits = props.evaluation?.colored_digits || [];
  return showLastThreshold.value ? digits.slice(-lastDigitCount.value) : [];
});
const leadingThresholded = computed(() => {
  const digits = tresholdedImages.value || [];
  return showLastThreshold.value ? digits.slice(0, -lastDigitCount.value) : digits;
});
const trailingThresholded = computed(() => {
  const digits = tresholdedImages.value || [];
  return showLastThreshold.value ? digits.slice(-lastDigitCount.value) : [];
});
const leadingDigitModels = computed(() => {
  const models = currentDigitModels.value || [];
  return showLastThreshold.value ? models.slice(0, -lastDigitCount.value) : models;
});
const trailingDigitModels = computed(() => {
  const models = currentDigitModels.value || [];
  return showLastThreshold.value ? models.slice(-lastDigitCount.value) : [];
});
const trailingOffset = computed(() => leadingDigitModels.value.length);
const digitWidth = computed(() => {
  const count = props.evaluation?.colored_digits?.length || props.segments || 1;
  const base = 250 / Math.max(count, 1);
  const width = Math.min(base, 48);
  return `${width}px`;
});

const updateThreshold = (value) => {
  currentThreshold.value = value;
};

const updateThresholdLast = (value) => {
  currentThresholdLast.value = value;
};

const updateIslandingPadding = (value) => {
  currentIslandingPadding.value = value;
};

const startThresholdSearch = () => {
  emits('searchThresholds', searchSteps.value);
};

onMounted(() => {
  refreshThresholds();
});

watch(() => props.evaluation, () => {
  refreshThresholds();
});

watch(() => props.threshold, (newVal) => {
  currentThreshold.value = newVal;
  refreshThresholds();
});

watch(() => props.threshold_last, (newVal) => {
  currentThresholdLast.value = newVal;
  refreshThresholds();
});

watch(() => props.islanding_padding, (newVal) => {
  currentIslandingPadding.value = newVal;
  refreshThresholds();
});

watch(() => props.segments, () => {
  refreshThresholds();
});

watch(() => props.decimals, (newVal) => {
  currentDecimals.value = Number.isFinite(newVal) ? newVal : 3;
  refreshThresholds();
});

watch(digitCount, (next) => {
  if (currentDecimals.value > next) {
    currentDecimals.value = next;
    emits('update-decimals', next);
  }
});

const sendUpdate = () => {
  emits('update', {
    threshold: currentThreshold.value,
    threshold_last: currentThresholdLast.value,
    islanding_padding: currentIslandingPadding.value,
  });
  refreshThresholds();
}

const normalizeDigitModels = (models, count) => {
  const normalized = [];
  for (let i = 0; i < count; i++) {
    const value = Array.isArray(models) ? models[i] : null;
    normalized.push(value === 'segment' ? 'segment' : 'rotating');
  }
  return normalized;
};

const syncDigitModels = () => {
  const count = digitCount.value;
  if (!count) {
    currentDigitModels.value = [];
    return;
  }
  currentDigitModels.value = normalizeDigitModels(props.digitModels, count);
};

const toggleDigitModel = (index) => {
  const models = [...currentDigitModels.value];
  const current = models[index] === 'segment' ? 'segment' : 'rotating';
  models[index] = current === 'segment' ? 'rotating' : 'segment';
  currentDigitModels.value = models;
  emits('update-digit-models', models);
};

const changeDecimals = (delta) => {
  const nextValue = Math.max(0, Math.min(digitCount.value, (currentDecimals.value || 0) + delta));
  currentDecimals.value = nextValue;
  emits('update-decimals', nextValue);
  refreshThresholds();
};

watch(
  () => [props.digitModels, props.evaluation, props.segments],
  () => {
    syncDigitModels();
  },
  { immediate: true }
);

const refreshThresholds = async () => {
  if (refreshing.value) return;
  if (props.loading) return;
  refreshing.value = true;

  let narray = [];
  const base64s = props.evaluation["colored_digits"];
  for (let j = 0; j < base64s.length; j++) {
    const isLast = showLastThreshold.value && j >= base64s.length - lastDigitCount.value;
    const threshold = isLast ? currentThresholdLast.value : currentThreshold.value;
    const newBase64 = await thresholdImage(base64s[j], threshold, currentIslandingPadding.value);
    narray.push(newBase64);
  }
  tresholdedImages.value = narray;
  refreshing.value = false;
}

const host = import.meta.env.VITE_HOST;

async function thresholdImage(base64, threshold, islanding_padding = 0) {
  // use endpoint /api/evaluate/single
  const response = await fetch(host + 'api/evaluate/single', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      base64str: base64,
      threshold_low: threshold[0],
      threshold_high: threshold[1],
      islanding_padding: islanding_padding,
      invert: true
    })
  });
  const result = await response.json();
  return result.base64;
}

</script>

<style scoped>
.digit{
  width: auto;
  height: 40px;
  object-fit: contain;
}

.digit-models-row :deep(.n-button) {
  padding: 0 6px;
}

.digit-models-title {
  font-size: 12px;
  opacity: 0.7;
  margin: 4px 0 8px;

}

.digit-model-button {
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.digit-model-button.segment {
  border-color: rgba(82, 196, 26, 0.7);
}

.digit-model-button.rotating {
  border-color: rgba(64, 158, 255, 0.7);
}

.digit-model-icon {
  width: 14px;
  height: 14px;
  display: block;
}

.decimal-divider {
  display: flex;
  align-items: center;
  gap: 0px;
  padding: 0 10px;
}

.decimal-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.7);
}

.light-mode .decimal-dot {
  background: rgba(0, 0, 0, 0.6);
}

.th {
  border: 1px solid rgba(255, 255, 255, 0.16);
  mix-blend-mode: screen;
}

.threshold-scroll {
  overflow-x: auto;
}

.card-title{
  text-transform: uppercase;
  width:100%;
  background-color: rgba(125, 125, 125, 0.1);
  text-align: center;
}

</style>
