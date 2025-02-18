<template>
  <n-flex size="large">
    <div  style="max-width: 300px">
      <n-card v-if="data" :title="new Date(data.picture.timestamp).toLocaleString()" size="small">
        <template #cover>
          <img :src="'data:image/jpeg;base64,' + data.picture.data" alt="Watermeter"/>
        </template>
      </n-card>
      <n-divider/>
      <n-button type="info" style="width: 100%">
        Enable Setup mode
      </n-button>
      <n-button type="error" style="width: 100%;margin-top: 5px">
        Delete
      </n-button>
    </div>
    <div style="padding-left: 20px">
      <n-h2>Details of {{id}}</n-h2>
      <template v-if="evaluations && evaluations.evals">
        <template v-for="[i, encoded] in evaluations.evals.entries()" :key="i">
          <n-flex>
            {{new Date(data.picture.timestamp).toLocaleString()}}
            <table >
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
          </n-flex>
          <n-divider/>
        </template>
      </template>
    </div>
  </n-flex>
</template>

<script setup>
import {NH2, NFlex, NCard, NDivider, NButton} from 'naive-ui';
import { useRoute } from 'vue-router';
import {onMounted, ref} from "vue";

const route = useRoute();

const id = route.params.id;

onMounted(() => {
  loadMeter();
});

const data = ref(null);
const evaluations = ref(null);

const loadMeter = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + '/api/watermeters/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  data.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + '/api/watermeters/' + id + '/evals', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  evaluations.value = await response.json();
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