<template>
  <n-flex size="large">
    <div  style="max-width: 300px">
      <n-card v-if="data" :title="new Date(data.picture.timestamp).toLocaleString()" size="small">
        <template #cover>
          <img :src="'data:image/jpeg;base64,' + data.picture.data" alt="Watermeter"/>
        </template>
      </n-card>
      <n-divider/>
      <n-popconfirm
        @positive-click="resetToSetup"
      >
        <template #trigger>
          <n-button type="info" style="width: 100%">
            Enable Setup mode
          </n-button>
        </template>
        This will reset the history of this meter and enable setup mode. Are you sure?
      </n-popconfirm>

      <n-button type="error" style="width: 100%;margin-top: 5px">
        Delete
      </n-button>
    </div>
    <div style="padding-left: 20px">
      <n-h2>Details of {{id}}</n-h2>
      <template v-if="evaluations && evaluations.evals">
        <template v-for="[i, encoded] in evaluations.evals.slice().reverse().entries()" :key="i">
          <n-flex>
            {{new Date(JSON.parse(encoded)[3]).toLocaleString()}}<br>
            {{ (JSON.parse(encoded)[2].reduce((partialSum, a) => partialSum * a[0][1], 1) * 100).toFixed(1) }}%<br>
            <table >
              <tr>
                <td v-for="base64 in JSON.parse(encoded)[1]" :key="base64">
                  <img class="digit" :src="'data:image/jpeg;base64,' + base64" alt="Watermeter"/>
                </td>
              </tr>
              <tr>
                <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">
                  <span class="prediction" >
                    {{ (digit[0][0]=='r')? '↕' : digit[0][0] }}
                  </span>
                </td>
              </tr>
              <tr>
                <td v-for="[i, digit] in JSON.parse(encoded)[2].entries()" :key="i + '' + i" style="text-align: center;">
                  <span class="confidence" :style="{color: getColor(digit[0][1])}">
                    {{ Math.round(digit[0][1] * 100) }}
                  </span>
                </td>
              </tr>
            </table>
          </n-flex>
          <n-divider/>
        </template>
      </template>
    </div>
    <div>
      <apex-chart width="500" type="line" :series="series" :options="options"></apex-chart>
    </div>
  </n-flex>
</template>

<script setup>
import {NH2, NFlex, NCard, NDivider, NButton, NPopconfirm} from "naive-ui";
import { useRoute } from 'vue-router';
import {computed, onMounted, ref} from "vue";
import router from "@/router";
import ApexChart from 'vue3-apexcharts';

const route = useRoute();

const id = route.params.id;

onMounted(() => {
  loadMeter();
});

const data = ref(null);
const evaluations = ref(null);
const history = ref(null);

function getColor(value) {
  // Clamp the value between 0 and 1
  value = Math.max(0, Math.min(1, value));

  // Map value (0.0 to 1.0) to hue (0 = red, 60 = yellow, 120 = green)
  const hue = value * 120;

  // Using 100% saturation and 40% lightness for good contrast on white.
  return `hsl(${hue}, 100%, 40%)`;
}

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

  response = await fetch(process.env.VUE_APP_HOST + '/api/watermeters/' + id + '/history', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  history.value = await response.json();
}

const series = computed(() => {
  if (history.value) {
    return [{
      name: 'Consumption m³',
      data: history.value.history.map((item) => [item[1], item[0] / 1000])
    }]
  } else {
    return [];
  }
});

const options = {
  chart: {
    type: 'line',
    zoom: {
      enabled: true
    }
  },
  xaxis: {
    type: 'datetime',
    labels: {
      format: 'dd MMM HH:mm'
    },
    title: {
      text: 'Time'
    }
  },
  yaxis: {
    title: {
      text: 'Consumption m³'
    }
  },
  stroke: {
    curve: 'smooth'
  },
  tooltip: {
    x: {
      format: 'dd MMM HH:mm'
    }
  }
};

const resetToSetup = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + '/api/setup/' + id + '/reset', {
    method: 'POST',
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });

  if (response.status === 200) {
    router.replace({path: '/setup/' + id});
  } else {
    console.log('Error resetting meter');
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
  font-size: 12px;
}
</style>