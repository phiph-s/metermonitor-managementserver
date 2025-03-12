<template>
  <router-link to="/"><n-button quaternary round type="primary" size="large" style="padding: 0; font-size: 16px;">
   ← Back
  </n-button></router-link><br><br>
  <n-flex size="large">
    <div  style="max-width: 300px">
      <n-card v-if="data" :title="new Date(data.picture.timestamp).toLocaleString()" size="small">
        <template #cover>
          <img :src="'data:image/'+data.picture.format+';base64,' + data.picture.data" alt="Watermeter" :class="{rotated: rotated180}"/>
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

      <n-popconfirm
        @positive-click="deleteMeter"
      >
        <template #trigger>
          <n-button type="error" style="width: 100%;margin-top: 5px">
            Delete
          </n-button>
        </template>
        This will delete the meter with all its settings and data. Are you sure?
      </n-popconfirm>

      <n-h6>
        Settings
      </n-h6>
      <b>
        Thresholds: {{threshold[0]}} - {{threshold[1]}}<br>
        Last digit thresholds: {{threshold_last[0]}} - {{threshold_last[1]}}<br>
        Islanding padding: {{islanding_padding}}<br>
        Segments: {{segments}}<br>
        Extended last digit: {{extendedLastDigit}}<br>
        Last 3 digits narrow: {{last3DigitsNarrow}}<br>
        Rotated 180: {{rotated180}}<br>
        Invert: {{invert}}
      </b>
    </div>
    <div style="padding-left: 20px;">
      <div style="height: calc(100vh - 120px); overflow: scroll;" class="bglight">
        <n-h2>Last evaluations of {{id}}</n-h2>
        <template v-if="decodedEvals">
      <template v-for="[i, evalDecoded] in decodedEvals.entries()" :key="i">
        <n-flex :class="{redbg: evalDecoded[4] == null}">
          <n-flex vertical>
            {{new Date(evalDecoded[3]).toLocaleString()}}<br>
            <div v-if="evalDecoded[6]" :style="{color: getColor(evalDecoded[6]), fontSize: '20px'}">
              <b>{{(evalDecoded[6] * 100).toFixed(1)}}</b>%
            </div>
          </n-flex>
          <table>
            <tr>
              <td v-for="base64 in evalDecoded[1]" :key="base64">
                <img class="digit" :src="'data:image/png;base64,' + base64" alt="Watermeter"/>
              </td>
            </tr>
            <tr>
              <td v-for="[i, digit] in evalDecoded[2].entries()" :key="i + 'v'" style="text-align: center;">
                <span class="prediction" >
                  {{ (digit[0][0]=='r')? '↕' : digit[0][0] }}
                </span>
              </td>
            </tr>
            <tr>
              <td v-for="[i, digit] in evalDecoded[2].entries()" :key="i + 'e'" style="text-align: center;">
                <span class="confidence" :style="{color: getColor(digit[0][1])}">
                  {{ Math.round(digit[0][1] * 100) }}
                </span>
              </td>
            </tr>
            <tr v-if="evalDecoded[5]">
              <td v-for="[i, digit] in evalDecoded[5].entries()" :key="i + 'g'" style="text-align: center;">
                <span class="prediction" v-if="digit !== evalDecoded[2][i][0][0]">
                  {{ digit?digit[0]:'' }}
                </span>
              </td>
            </tr>
            <tr v-if="evalDecoded[5]">
              <td v-for="[i, digit] in evalDecoded[5].entries()" :key="i + 'h'" style="text-align: center;">
                <span class="confidence" :style="{color: getColor(digit?digit[1]:0)}">
                  {{ digit?digit[1]:'' }}
                </span>
              </td>
            </tr>
            <tr v-if="evalDecoded[4]">
              <td v-for="[i, digit] in (evalDecoded[4] + '').padStart(evalDecoded[1].length, '0').split('').entries()" :key="i + 'f'" style="text-align: center;">
                <span class="prediction red" v-if="digit !== evalDecoded[2][i][0][0]">
                  {{ digit }}
                </span>
              </td>
            </tr>
          </table>
        </n-flex>
        <n-divider/>
      </template>
    </template>
      </div>
    </div>
    <div>
      <apex-chart class="bg" width="500" type="line" :series="series" :options="options"></apex-chart>
      <apex-chart class="bg" width="500" type="line" :series="seriesConf" :options="optionsConf"></apex-chart>
    </div>
  </n-flex>
</template>

<script setup>
import {NH2, NH6, NFlex, NCard, NDivider, NButton, NPopconfirm} from "naive-ui";
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

const threshold = ref([0, 0]);
const threshold_last = ref([0, 0]);
const islanding_padding = ref(0);
const segments = ref(0);
const extendedLastDigit = ref(false);
const last3DigitsNarrow = ref(false);
const rotated180 = ref(false);
const invert = ref(false);

const decodedEvals = computed(
  () => {
    console.log(evaluations.value ? evaluations.value.evals.map((encoded) => JSON.parse(encoded)).reverse() : [])
    return evaluations.value ? evaluations.value.evals.map((encoded) => JSON.parse(encoded)).reverse() : []
  }
);
function getColor(value) {
  // Clamp the value between 0 and 1
  value = Math.max(0, Math.min(1, value));

  // Map value (0.0 to 1.0) to hue (0 = red, 60 = yellow, 120 = green)
  const hue = value * 120;

  // Using 100% saturation and 40% lightness for good contrast on white.
  return `hsl(${hue}, 100%, 40%)`;
}

const loadMeter = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  data.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id + '/evals', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  evaluations.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id + '/history', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  history.value = await response.json();

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

const series = computed(() => {
  if (history.value) {
    return [{
      name: 'Consumption m³',
      data: history.value.history.map((item) => [new Date(item[1]), item[0] / 1000])
    }]
  } else {
    return [];
  }
});

const seriesConf = computed(() => {
  if (history.value) {
    console.log(history.value)
    return [{
      name: 'Confidence in %',
      data: history.value.history.map((item) => [new Date(item[1]), item[2] * 100])
    }]
  } else {
    return [];
  }
});

const options = {
  title: {
    text: 'Consumption'
  },
  chart: {
    type: 'line',
    zoom: {
      enabled: true
    }
  },
  xaxis: {
    type: 'datetime',
    labels: {
      formatter: function (value, timestamp) {
      return new Date(timestamp).toLocaleString() // The formatter function overrides format property
    },
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

const optionsConf = {
  title: {
    text: 'Confidence'
  },
  chart: {
    type: 'line',
    zoom: {
      enabled: true
    }
  },
  xaxis: {
    type: 'datetime',
    labels: {
      formatter: function (value, timestamp) {
      return new Date(timestamp).toLocaleString() // The formatter function overrides format property
    },
    },
    title: {
      text: 'Time'
    }
  },
  yaxis: {
    title: {
      text: 'Confidence in %'
    },
    labels: {
      formatter: function (value) {
        return value.toFixed(1) + '%';
      }
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

const deleteMeter = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + 'api/watermeters/' + id, {
    method: 'DELETE',
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });

  if (response.status === 200) {
    router.replace({path: '/'});
  } else {
    console.log('Error deleting meter');
  }
}

const resetToSetup = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + 'api/setup/' + id + '/reset', {
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

.red{
  color: red;
}

.redbg{
  background-color: rgba(255, 0, 0, 0.1);
}

.confidence{
  margin: 3px;
  font-size: 12px;
}

.bg{
  background-color: rgba(240, 240, 240, 0.8);
  padding: 20px;
}

.bglight{
  background-color: rgba(240, 240, 240, 0.1);
  padding: 20px;
}

.rotated{
  transform: rotate(180deg);
}
</style>