<template>
  <h2>Setup for {{ id }}</h2>

  <img v-if="lastPicture" :src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data" alt="Watermeter"/>
  <br>
  <label>
    <input type="number" v-model="segments"/>
    Segments
  </label><br>
  <label>
    <input type="checkbox" v-model="extendedLastDigit"/>
    Extended last digit
  </label><br>
  <label>
    <input type="checkbox" v-model="last3DigitsNarrow"/>
    Last 3 digits are narrow
  </label>
  <hr>
  <table>
    <tr>
      <td>Unprocessed</td>
      <td>Thresholds applied</td>
    </tr>
    <tr>
      <td>
        <template v-for="encoded in evaluations.evals" :key="encoded">
            <img class="digit" v-for="base64 in JSON.parse(encoded)[0]" :src="'data:image/jpeg;base64,' + base64" :key="base64" alt="Watermeter"/>
            <br>
        </template>
      </td>
      <td>
        <template v-for="run in tresholdedImages" :key="run">
            <img class="digit" v-for="base64 in run" :src="'data:image/jpeg;base64,' + base64" :key="base64" alt="Watermeter"/>
            <br>
        </template>
      </td>
    </tr>
  </table>

  <label>
    <input type="range" v-model="threshold_low" min="0" max="255" />
    Low Threshold {{threshold_low}}
  </label><br>
  <label>
    <input type="range" v-model="threshold_high" min="0" max="255" />
    High Threshold {{threshold_high}}
  </label><br>
  <label>
    <input type="checkbox" v-model="invert"/>
    Invert colors
  </label>

  <hr>
  Historic Evaluations

  <table v-for="[i, encoded] in evaluations.evals.entries()" :key="i">
    <tr>
      <td v-for="base64 in JSON.parse(encoded)[1]" :key="base64">
        <img class="digit" :src="'data:image/jpeg;base64,' + base64" alt="Watermeter"/>
      </td>
    </tr>
    <tr>
      <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">
        <span class="prediction" >
          {{ digit[0] }}
        </span>
      </td>
    </tr>
    <tr>
      <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">
        <span class="confidence" >
          {{ (digit[1] * 100).toFixed(2) }}
        </span>
      </td>
    </tr>
  </table>

</template>

<script setup>
import {onMounted, ref, watch} from 'vue';
import router from "@/router";
import { useRoute } from 'vue-router';

const route = useRoute();
const id = route.params.id;

const lastPicture = ref("");
const evaluations = ref("");

const threshold_low = ref(0);
const threshold_high = ref(0);
const tresholdedImages = ref([]);

const segments = ref(0);
const extendedLastDigit = ref(false);
const last3DigitsNarrow = ref(false);
const invert = ref(false);

const getData = async () => {
  let response = await fetch('/api/watermeters/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  lastPicture.value = await response.json();

  response = await fetch('/api/watermeters/' + id + '/evals', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  evaluations.value = await response.json();

  response = await fetch('/api/settings/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });

  let result = await response.json();
  threshold_low.value = result.threshold_low + "";
  threshold_high.value = result.threshold_high + "";
  segments.value = result.segments;
  extendedLastDigit.value = result.extended_last_digit === 1;
  last3DigitsNarrow.value = result.shrink_last_3 === 1;
  invert.value = result.invert === 1;

  refreshThresholds({low: threshold_low.value, high: threshold_high.value}, false);

}

onMounted(() => {
  // check if secret is in local storage
  const secret = localStorage.getItem('secret');
  if (secret === null) {
    router.push({ path: '/unlock' });
  }
  getData();
});

watch([threshold_low, threshold_high, invert], () => {
  refreshThresholds({low: threshold_low.value, high: threshold_high.value});
});

watch([segments, extendedLastDigit, last3DigitsNarrow], () => {
  updateSettings();
});

let refreshing = false;

const refreshThresholds = async (newThresholds, update = true) => {
  if (refreshing) return;
  threshold_low.value = newThresholds.low;
  threshold_high.value = newThresholds.high;
  refreshing = true;

  tresholdedImages.value = [];
  for (let i = 0; i < evaluations.value.evals.length; i++) {
    const encoded = evaluations.value.evals[i];
    const base64s = JSON.parse(encoded)[0];
    const newBase64s = [];
    for (let j = 0; j < base64s.length; j++) {
      const newBase64 = await thresholdImage(base64s[j], newThresholds.low, newThresholds.high);
      newBase64s.push(newBase64);
    }
    tresholdedImages.value.push(newBase64s);
  }

  refreshing = false;
  if (update) updateSettings();
}

const updateSettings = async () => {

  const settings = {
    name: id,
    threshold_low: threshold_low.value,
    threshold_high: threshold_high.value,
    segments: segments.value,
    extended_last_digit: extendedLastDigit.value,
    shrink_last_3: last3DigitsNarrow.value,
    invert: invert.value
  }

  await fetch('/api/settings', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(settings)
  });
}

async function thresholdImage(base64, thresholdLow, thresholdHigh) {
  // use endpoint /api/evaluate/single
  const response = await fetch('/api/evaluate/single', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      base64str: base64,
      threshold_low: thresholdLow,
      threshold_high: thresholdHigh,
      invert: invert.value
    })
  });
  const result = await response.json();
  return result.base64;
}

</script>

<style scoped>
.digit{
  margin: 3px;
}

.prediction{
  margin: 3px;
  font-size: 20px;
}

.confidence{
  margin: 3px;
  font-size: 10px;
}
</style>