<template>
  <template v-if="latestEval">
    <n-card>
      <n-flex justify="space-around" size="large">
        <img class="digit" v-for="[i,base64] in JSON.parse(latestEval)[1].entries()" :key="i + 'c'" :src="'data:image/jpeg;base64,' + base64" alt="D"/>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="prediction" v-for="[i, digit] in JSON.parse(latestEval)[2].entries()" :key="i + 'd'">
          {{ digit[0] }}
        </span>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="confidence" v-for="[i, digit] in JSON.parse(latestEval)[2].entries()" :key="i + 'e'">
          {{ (digit[1] * 100).toFixed(2) }}
        </span>
      </n-flex>
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
import { defineProps } from 'vue';
import {NFlex, NCard, NButton} from "naive-ui";
import router from "@/router";

const props = defineProps([
    'meterid',
    'latestEval'
]);

const finishSetup = async () => {
  // post to /api/setup/{name}/finish
  const r = await fetch(process.env.VUE_APP_HOST + '/api/setup/' + props.meterid + '/finish', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });

  if (r.status === 200) {
    router.push({ path: '/' });
  } else {
    console.log('Error finishing setup');
  }
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