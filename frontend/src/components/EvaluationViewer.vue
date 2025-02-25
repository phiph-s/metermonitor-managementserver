<template>
  <template v-if="latestEval">
    <n-card>
      <n-flex justify="space-around" size="large">
        <img class="digit" v-for="[i,base64] in JSON.parse(latestEval)[1].entries()" :key="i + 'c'" :src="'data:image/jpeg;base64,' + base64" alt="D"/>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="prediction" v-for="[i, digit] in JSON.parse(latestEval)[2].entries()" :key="i + 'd'">
          {{ (digit[0][0]=='r')? '↕' : digit[0][0] }}
        </span>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="confidence" v-for="[i, digit] in JSON.parse(latestEval)[2].entries()" :key="i + 'e'" :style="{color: getColor(digit[0][1])}">
          {{ (digit[0][1] * 100).toFixed(2) }}
        </span>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="prediction" v-for="[i, digit] in JSON.parse(latestEval)[5].entries()" :key="i + 'c'">
          {{ digit?digit[0]:'' }}
        </span>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="confidence" v-for="[i, digit] in JSON.parse(latestEval)[5].entries()" :key="i + 'c'" :style="{color: getColor(digit?digit[1]:0)}">
          {{ digit?digit[1]:'' }}
        </span>
      </n-flex>
      <n-divider />
      <n-h3>Manual initial read</n-h3>
      {{new Date(timestamp).toLocaleString()}}<br>
      <n-input-number v-model:value="initialValue" placeholder="Readout" />
      <template #action>
        <n-flex justify="end" size="large">
          <n-button
              @click="finishSetup"
              type="primary"
          >Finish & save</n-button>
        </n-flex>
      </template>
    </n-card>
  </template>
</template>

<script setup>
import {defineProps, ref} from 'vue';
import {NFlex, NCard, NButton, NInputNumber, NDivider, NH3} from 'naive-ui';
import router from "@/router";

const props = defineProps([
    'meterid',
    'latestEval',
    'timestamp'
]);

const initialValue = ref(0);

const finishSetup = async () => {
  // post to /api/setup/{name}/finish
  const r = await fetch(process.env.VUE_APP_HOST + 'api/setup/' + props.meterid + '/finish', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      'value': initialValue.value,
      'timestamp': props.timestamp
    })
  });

  if (r.status === 200) {
    router.push({ path: '/' });
  } else {
    console.log('Error finishing setup');
  }
}

function getColor(value) {
  // Clamp the value between 0 and 1
  value = Math.max(0, Math.min(1, value));

  // Map value (0.0 to 1.0) to hue (0 = red, 60 = yellow, 120 = green)
  const hue = value * 120;

  // Using 100% saturation and 40% lightness for good contrast on white.
  return `hsl(${hue}, 100%, 40%)`;
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