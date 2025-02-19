<template>
  <n-card>
    <n-flex justify="space-around" size="large" v-if="encoded">
      <img class="digit" v-for="[i,base64] in JSON.parse(encoded)[0].entries()" :src="'data:image/jpeg;base64,' + base64" :key="i+'a'" alt="D" style="max-width: 40px"/>
    </n-flex>
    <n-flex justify="space-around" size="large" v-if="tresholdedImages">
      <img class="digit" v-for="[i,base64] in tresholdedImages.entries()" :src="'data:image/jpeg;base64,' + base64" :key="i+'b'" alt="Watermeter"/>
    </n-flex>

    <n-divider dashed></n-divider>

    <label>
      <input type="range" v-model="nthreshold_low" min="0" max="255" @mouseup="sendUpdate" />
      Low Threshold {{nthreshold_low}}
    </label><br>
    <label>
      <input type="range" v-model="nthreshold_high" min="0" max="255" @mouseup="sendUpdate" />
      High Threshold {{nthreshold_high}}
    </label><br>
    <label>
      <input type="checkbox" v-model="ninvert"/>
      Invert colors
    </label>
    <template #action>
      <n-flex justify="end" size="large">
        <n-button
            @click="emits('reevaluate')"
            type="primary"
        >(Re)evaluate</n-button>
      </n-flex>
    </template>
  </n-card>
</template>

<script setup>
import {NFlex, NCard, NDivider, NButton} from "naive-ui";
import {defineProps, defineEmits, ref, watch} from 'vue';

const props = defineProps([
    'encoded',
    'threshold_low',
    'threshold_high',
    'invert'
]);

const emits = defineEmits(['update', 'reevaluate']);

const nthreshold_low = ref(props.threshold_low);
const nthreshold_high = ref(props.threshold_high);
const ninvert = ref(props.invert);
const tresholdedImages = ref([]);
const refreshing = ref(false);

watch(() => props.threshold_low, (newVal) => {
  nthreshold_low.value = newVal;
});
watch(() => props.threshold_high, (newVal) => {
  nthreshold_high.value = newVal;
});
watch(() => props.invert, (newVal) => {
  ninvert.value = newVal;
});

const sendUpdate = () => {
  emits('update', {
    threshold_low: nthreshold_low.value,
    threshold_high: nthreshold_high.value,
    invert: ninvert.value
  });
  refreshThresholds();
}

watch([ninvert], () => {
  sendUpdate();
});

const refreshThresholds = async () => {
  if (refreshing.value) return;
  refreshing.value = true;

  let narray = [];
  const base64s = JSON.parse(props.encoded)[0];
  for (let j = 0; j < base64s.length; j++) {
    const newBase64 = await thresholdImage(base64s[j], nthreshold_low.value, nthreshold_high.value);
    narray.push(newBase64);
  }
  tresholdedImages.value = narray;
  refreshing.value = false;
}

async function thresholdImage(base64, thresholdLow, thresholdHigh) {
  // use endpoint /api/evaluate/single
  const response = await fetch(process.env.VUE_APP_HOST + '/api/evaluate/single', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      base64str: base64,
      threshold_low: thresholdLow,
      threshold_high: thresholdHigh,
      invert: ninvert.value
    })
  });
  const result = await response.json();
  return result.base64;
}

</script>

<style scoped>

</style>