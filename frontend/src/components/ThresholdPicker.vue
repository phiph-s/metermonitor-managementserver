<template>
  <n-card size="small">
    <template #cover>
      <div class="card-title">
        <span style="font-weight: bolder;">Select thresholds</span><span style="opacity: 0.7"></span>
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

    <n-flex :size="[0,0]" justify="space-around" align="center">
      <div>
        <n-flex justify="space-around" size="large" v-if="evaluation">
          <img :style="{ width: digitWidth }" class="digit" v-for="[i,base64] in leadingDigits.entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" />
        </n-flex>
        <br>
        <n-flex justify="space-around" size="large" v-if="tresholdedImages" class="theme-revert">
          <img :style="{ width: digitWidth }" class="digit th" v-for="[i,base64] in leadingThresholded.entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter" />
        </n-flex>
        <br>
        <n-slider :value="currentThreshold" @update:value="updateThreshold" range :step="1" :max="255" @mouseup="sendUpdate" style="max-width: 150px;" :disabled="isDisabled"/>
        {{currentThreshold[0]}} - {{currentThreshold[1]}}
      </div>
      <n-divider vertical v-if="showLastThreshold" />
      <div v-if="showLastThreshold">
        <n-flex justify="space-around" size="large" v-if="evaluation">
          <img :style="{ width: digitWidth }" class="digit" v-for="[i,base64] in trailingDigits.entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" />
        </n-flex>
        <br>
        <n-flex justify="space-around" size="large" v-if="tresholdedImages" class="theme-revert">
          <img :style="{ width: digitWidth }" class="digit th" v-for="[i,base64] in trailingThresholded.entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter" />
        </n-flex>
        <br>
        <n-slider :value="currentThresholdLast" @update:value="updateThresholdLast" range :step="1" :max="255" @mouseup="sendUpdate" style="max-width: 150px;" :disabled="isDisabled"/>
        {{currentThresholdLast[0]}} - {{currentThresholdLast[1]}}
      </div>
    </n-flex>

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
import {NFlex, NCard, NDivider, NButton, NSlider, NInputNumber, NSpin, NInputGroup, NInputGroupLabel} from 'naive-ui';
import {defineProps, defineEmits, ref, watch, onMounted, computed} from 'vue';

const props = defineProps([
    'evaluation',
    'threshold',
    'threshold_last',
    'islanding_padding',
    'segments',
    'loading',
    'searchingThresholds',
    'thresholdSearchResult'
]);

const emits = defineEmits(['update', 'reevaluate', 'next', 'searchThresholds']);

const currentThreshold = ref(props.threshold);
const currentThresholdLast = ref(props.threshold_last);
const currentIslandingPadding = ref(props.islanding_padding);

const tresholdedImages = ref([]);
const refreshing = ref(false);
const searchSteps = ref(10);

const isDisabled = computed(() => props.loading || props.searchingThresholds);
const segmentCount = computed(() => {
  const evaluationCount = props.evaluation?.colored_digits?.length || 0;
  const value = props.segments || evaluationCount;
  return value || evaluationCount || 0;
});
const showLastThreshold = computed(() => segmentCount.value > 3);
const leadingDigits = computed(() => {
  const digits = props.evaluation?.colored_digits || [];
  return showLastThreshold.value ? digits.slice(0, -3) : digits;
});
const trailingDigits = computed(() => {
  const digits = props.evaluation?.colored_digits || [];
  return showLastThreshold.value ? digits.slice(-3) : [];
});
const leadingThresholded = computed(() => {
  const digits = tresholdedImages.value || [];
  return showLastThreshold.value ? digits.slice(0, -3) : digits;
});
const trailingThresholded = computed(() => {
  const digits = tresholdedImages.value || [];
  return showLastThreshold.value ? digits.slice(-3) : [];
});
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

const sendUpdate = () => {
  emits('update', {
    threshold: currentThreshold.value,
    threshold_last: currentThresholdLast.value,
    islanding_padding: currentIslandingPadding.value,
  });
  refreshThresholds();
}

const refreshThresholds = async () => {
  if (refreshing.value) return;
  if (props.loading) return;
  refreshing.value = true;

  let narray = [];
  const base64s = props.evaluation["colored_digits"];
  for (let j = 0; j < base64s.length; j++) {
    const isLast3 = showLastThreshold.value && j >= base64s.length - 3;
    const threshold = isLast3 ? currentThresholdLast.value : currentThreshold.value;
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
  width: 18px;
  height: auto;
}

.th {
  border: 1px solid rgba(255, 255, 255, 0.16);
  mix-blend-mode: screen;
}

.card-title{
  text-transform: uppercase;
  width:100%;
  background-color: rgba(125, 125, 125, 0.1);
  text-align: center;
}

</style>
