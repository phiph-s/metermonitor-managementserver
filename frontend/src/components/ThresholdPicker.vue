<template>
  <n-card>
    <n-flex>
      <div>
        <n-flex justify="space-around" size="large" v-if="encoded">
          <img class="digit" v-for="[i,base64] in JSON.parse(encoded)[0].slice(0,-3).entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" style="max-width: 40px"/>
        </n-flex>
        <n-flex justify="space-around" size="large" v-if="tresholdedImages">
          <img class="digit" v-for="[i,base64] in tresholdedImages.slice(0,-3).entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter"/>
        </n-flex>

        <n-divider dashed></n-divider>
        Thresholds
        <n-slider v-model:value="nthreshold" range :step="1" :max="255" @mouseup="sendUpdate" style="max-width: 150px;"/>
      </div>
      <div>
        <n-flex justify="space-around" size="large" v-if="encoded">
          <img class="digit" v-for="[i,base64] in JSON.parse(encoded)[0].slice(-3).entries()" :src="'data:image/png;base64,' + base64" :key="i+'a'" alt="D" style="max-width: 40px"/>
        </n-flex>
        <n-flex justify="space-around" size="large" v-if="tresholdedImages">
          <img class="digit" v-for="[i,base64] in tresholdedImages.slice(-3).entries()" :src="'data:image/png;base64,' + base64" :key="i+'b'" alt="Watermeter"/>
        </n-flex>

        <n-divider dashed></n-divider>
        Thresholds (last 3)
        <n-slider v-model:value="nthreshold_last" range :step="1" :max="255" @mouseup="sendUpdate" style="max-width: 150px;"/>
      </div>
    </n-flex>
    <n-divider></n-divider>
    <label>
      <input type="checkbox" v-model="ninvert"/>
      Invert colors
    </label><br><br>
    Extraction padding
      <n-slider v-model:value="islanding_padding" :step="1" :max="100" @mouseup="sendUpdate" style="max-width: 150px;"/>
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
import {NFlex, NCard, NDivider, NButton, NSlider} from "naive-ui";
import {defineProps, defineEmits, ref, watch} from 'vue';

const props = defineProps([
    'encoded',
    'threshold',
    'threshold_last',
    'islanding_padding',
    'invert'
]);

const emits = defineEmits(['update', 'reevaluate']);

const nthreshold = ref(props.threshold);
const nthreshold_last = ref(props.threshold_last);
const islanding_padding = ref(props.islanding_padding);

const ninvert = ref(props.invert);
const tresholdedImages = ref([]);
const refreshing = ref(false);



watch(() => props.threshold, (newVal) => {
  nthreshold.value = newVal;
});
watch(() => props.threshold_last, (newVal) => {
  nthreshold_last.value = newVal;
});
watch(() => props.islanding_padding, (newVal) => {
  islanding_padding.value = newVal;
});

watch(() => props.invert, (newVal) => {
  ninvert.value = newVal;
});

const sendUpdate = () => {
  emits('update', {
    threshold: nthreshold.value,
    threshold_last: nthreshold_last.value,
    islanding_padding: islanding_padding.value,
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
    let isLast3 = j >= base64s.length - 3;
    const newBase64 = await thresholdImage(base64s[j], isLast3? nthreshold_last.value : nthreshold.value, islanding_padding.value);
    narray.push(newBase64);
  }
  tresholdedImages.value = narray;
  refreshing.value = false;
}

async function thresholdImage(base64, threshold, islanding_padding = 0) {
  // use endpoint /api/evaluate/single
  const response = await fetch(process.env.VUE_APP_HOST + 'api/evaluate/single', {
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
      invert: ninvert.value
    })
  });
  const result = await response.json();
  return result.base64;
}

</script>

<style scoped>

</style>