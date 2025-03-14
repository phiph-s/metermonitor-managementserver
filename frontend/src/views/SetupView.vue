<template>
  <router-link to="/"><n-button quaternary round type="primary" size="large" style="padding: 0; font-size: 16px;">
     ← Back
  </n-button></router-link>

  <h2>Setup for {{ id }}</h2>

  <n-steps :current="current" :status="currentStatus">
    <n-step
      title="Extraction/Segmentation"
    >
      <SegmentationConfigurator
          :last-picture="lastPicture"
          :extended-last-digit="extendedLastDigit"
          :last-3-digits-narrow="last3DigitsNarrow"
          :segments="segments"
          :rotated180="rotated180"
          :encoded-latest="evaluations.evals?evaluations.evals[evaluations.evals.length-1]:null"
          @update="updateSegmentationSettings"/>
    </n-step>
    <n-step
      title="Pick Thresholds"
    >
      <ThresholdPicker
          :encoded="evaluations.evals?evaluations.evals[evaluations.evals.length-1]:null"
          :run="tresholdedImages[tresholdedImages.length-1]"
          :invert="invert"
          :threshold="threshold"
          :threshold_last="threshold_last"
          :islanding_padding="islanding_padding"
          @update="updateThresholds"
          @reevaluate="reevaluate"
      />
    </n-step>
    <n-step
      title="Evaluation"
      v-if="lastPicture"
    >
      <EvaluationViewer :latest-eval="evaluations.evals?evaluations.evals[evaluations.evals.length-1]:null" :meterid="id" :timestamp="lastPicture.picture.timestamp"/>
    </n-step>
  </n-steps>
<!--  <table v-if="evaluations.evals">-->
<!--    <tr>-->
<!--      <td>Unprocessed</td>-->
<!--      <td>Thresholds applied</td>-->
<!--    </tr>-->
<!--    <tr>-->
<!--      <td>-->
<!--        <template v-for="encoded in evaluations.evals" :key="encoded">-->
<!--            <img class="digit" v-for="base64 in JSON.parse(encoded)[0]" :src="'data:image/jpeg;base64,' + base64" :key="base64" alt="Watermeter"/>-->
<!--            <br>-->
<!--        </template>-->
<!--      </td>-->
<!--      <td>-->
<!--        <template v-for="run in tresholdedImages" :key="run">-->
<!--            <img class="digit" v-for="base64 in run" :src="'data:image/jpeg;base64,' + base64" :key="base64" alt="Watermeter"/>-->
<!--            <br>-->
<!--        </template>-->
<!--      </td>-->
<!--    </tr>-->
<!--  </table>-->

<!--  <hr>-->
<!--  Historic Evaluations-->

<!--  <template v-if="evaluations.evals">-->
<!--    <table v-for="[i, encoded] in evaluations.evals.entries()" :key="i">-->
<!--      <tr>-->
<!--        <td v-for="base64 in JSON.parse(encoded)[1]" :key="base64">-->
<!--          <img class="digit" :src="'data:image/jpeg;base64,' + base64" alt="Watermeter"/>-->
<!--        </td>-->
<!--      </tr>-->
<!--      <tr>-->
<!--        <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">-->
<!--          <span class="prediction" >-->
<!--            {{ digit[0] }}-->
<!--          </span>-->
<!--        </td>-->
<!--      </tr>-->
<!--      <tr>-->
<!--        <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">-->
<!--          <span class="confidence" >-->
<!--            {{ (digit[1] * 100).toFixed(2) }}-->
<!--          </span>-->
<!--        </td>-->
<!--      </tr>-->
<!--    </table>-->
<!--  </template>-->

</template>

<script setup>
import {onMounted, ref} from 'vue';
import router from "@/router";
import { NSteps, NStep, NButton } from 'naive-ui';
import { useRoute } from 'vue-router';
import SegmentationConfigurator from "@/components/SegmentationConfigurator.vue";
import ThresholdPicker from "@/components/ThresholdPicker.vue";
import EvaluationViewer from "@/components/EvaluationViewer.vue";

const route = useRoute();
const id = route.params.id;

const lastPicture = ref("");
const evaluations = ref("");

const threshold = ref([0, 100]);
const threshold_last = ref([0, 100]);
const islanding_padding = ref(0);

const tresholdedImages = ref([]);

const segments = ref(0);
const extendedLastDigit = ref(false);
const last3DigitsNarrow = ref(false);
const rotated180 = ref(false);
const invert = ref(false);

const getData = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  lastPicture.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id + '/evals', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  evaluations.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + 'api/settings/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });

  let result = await response.json();

  threshold.value = [result.threshold_low, result.threshold_high];
  threshold_last.value = [result.threshold_last_low, result.threshold_last_high];
  islanding_padding.value = result.islanding_padding;

  segments.value = result.segments;
  extendedLastDigit.value = result.extended_last_digit === 1;
  last3DigitsNarrow.value = result.shrink_last_3 === 1;
  rotated180.value = result.rotated_180 === 1;
  invert.value = result.invert === 1;
}

onMounted(() => {
  // check if secret is in local storage
  const secret = localStorage.getItem('secret');
  if (secret === null) {
    router.push({ path: '/unlock' });
  }
  getData();
});

const updateThresholds = async (data) => {
  threshold.value = data.threshold;
  threshold_last.value = data.threshold_last;
  islanding_padding.value = data.islanding_padding;
  invert.value = data.invert;

  updateSettings();
}

const reevaluate = async () => {
  await fetch(process.env.VUE_APP_HOST + 'api/reevaluate_latest/' + id, {
    method: 'GET',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
    },
  });
  getData();
}

const updateSegmentationSettings = async (data) => {
  segments.value = data.segments;
  extendedLastDigit.value = data.extendedLastDigit;
  last3DigitsNarrow.value = data.last3DigitsNarrow;
  rotated180.value = data.rotated180;

  await updateSettings();
  reevaluate();
}
const updateSettings = async () => {

  const settings = {
    name: id,
    threshold_low: threshold.value[0],
    threshold_high: threshold.value[1],
    threshold_last_low: threshold_last.value[0],
    threshold_last_high: threshold_last.value[1],
    islanding_padding: islanding_padding.value,
    rotated_180: rotated180.value,
    segments: segments.value,
    extended_last_digit: extendedLastDigit.value,
    shrink_last_3: last3DigitsNarrow.value,
    invert: invert.value
  }

  await fetch(process.env.VUE_APP_HOST + 'api/settings', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(settings)
  });
}



</script>

<style scoped>
</style>