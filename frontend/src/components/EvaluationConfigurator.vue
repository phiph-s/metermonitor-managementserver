<template>
  <template v-if="evaluation">
    <n-card size="small">
      <template #cover>
        <div class="card-title">
          <span style="font-weight: bolder;">Result preview</span>
        </div>
      </template>
      <n-flex justify="space-around" size="large" style="padding-top: 16px;">
        <img :style="`width:calc(250px / ${evaluation['colored_digits'].length});`" class="digit theme-revert th" v-for="[i,base64] in evaluation['th_digits'].entries()" :key="i + 'c'" :src="'data:image/png;base64,' + base64" alt="D"/>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="prediction google-sans-code" v-for="[i, digit] in evaluation['predictions'].entries()" :key="i + 'd'">
          {{ (digit[0][0]==='r')? '↕' : digit[0][0] }}
        </span>
      </n-flex>
      <n-flex justify="space-around" size="large">
        <span class="confidence google-sans-code" v-for="[i, digit] in evaluation['predictions'].entries()" :key="i + 'e'" :style="{color: getColor(digit[0][1])}">
          {{ (digit[0][1] * 100).toFixed(0) }}
        </span>
      </n-flex><br>
    </n-card><br>
    <n-card size="small">
      <template #cover>
        <div class="card-title">
          <span style="font-weight: bolder;">Evaluation Settings</span><span style="opacity: 0.7"></span>
        </div>
      </template>
      <div style="max-width: 30%; padding-top: 16px;">
          <n-tooltip>
            <template #trigger>
              Correctional alg.
            </template>
            <span>
              Full: Positive flow check, max flow rate, fallback handling<br>
              Light: Only rotation and low-confidence digits corrected
            </span>
          </n-tooltip>
          <n-switch v-model:value="useCorrectionAlg" @update:value="onSwitchUpdate" :disabled="loading">
            <template #checked>Full</template>
            <template #unchecked>Light</template>
          </n-switch>
        </div>
      <n-flex style="padding-top: 8px;">
        <div style="max-width: 30%;">
            <n-tooltip>
              <template #trigger>
                Conf. threshold
              </template>
              <span>
                Set a confidence threshold for accepting digit predictions.<br>
                Digits with confidence below this value will be marked as uncertain.
              </span>
            </n-tooltip>
          <n-input-number ref="confThreshInput" :value="confidenceThreshold" @update:value="(e) => emit('update-conf-threshold', e)"
                          placeholder="e.g. 70%" :disabled="loading" />
        </div>
        <div style="max-width: 33%">
          Read initial value
          <n-input-number v-model:value="initialValue" placeholder="Readout" :disabled="loading || !useCorrectionAlg" />
        </div>
        <div style="max-width: 30%">
          Max. flow per hour
          <n-input-number v-model:value="maxFlowRateLocal"
                          @update:value="emit('update-max-flow', $event)"
                          placeholder="Flow rate" :disabled="loading || !useCorrectionAlg" />
        </div>
      </n-flex>
      <template #action>
        <n-flex justify="end" size="large">
          <n-button
              @click="finishSetup"
              type="success"
              round
              :disabled="loading"
              :loading="loading"
          >Finish & save</n-button>
        </n-flex>
      </template>
    </n-card>
  </template>
</template>

<script setup>
import {defineProps, ref, defineEmits, watch} from 'vue';
import {
  NFlex,
  NCard,
  NButton,
  NInputNumber,
  useDialog,
  NTooltip,
  NSwitch
} from 'naive-ui';
import router from "@/router";

const emit = defineEmits(['set-loading', 'update-max-flow', 'update-conf-threshold', 'update-use-correction']);

const props = defineProps([
    'meterid',
    'evaluation',
    'timestamp',
    'maxFlowRate',
    'confidenceThreshold',
    'useCorrectionAlg',
    'loading',
    'onSetLoading',
]);

const useCorrectionAlg = ref(props.useCorrectionAlg ?? true);

const predictionToValue = (predictions) =>
  predictions.reduce((acc, digit) => {
    const d = digit[0][0];
    return acc * 10 + (d === 'r' ? 0 : parseInt(d));
  }, 0);

const initialValue = ref(predictionToValue(props.evaluation['predictions']));

watch(() => props.evaluation?.predictions, (predictions) => {
  if (predictions) {
    initialValue.value = predictionToValue(predictions);
  }
});

const onSwitchUpdate = (val) => {
  if (!val) {
    // Switching to Light: fill empty fields with defaults
    if (!maxFlowRateLocal.value) {
      maxFlowRateLocal.value = 1;
      emit('update-max-flow', 1);
    }
    if (!initialValue.value) {
      initialValue.value = predictionToValue(props.evaluation['predictions']);
    }
  }
  useCorrectionAlg.value = val;
  emit('update-use-correction', val);
};

const maxFlowRateLocal = ref(props.maxFlowRate);

watch(() => props.maxFlowRate, (val) => { maxFlowRateLocal.value = val; });

const dialog = useDialog();
const host = import.meta.env.VITE_HOST;

const confThreshInput = ref(null);

const doFinishSetup = async () => {
  // notify parent to show loading
  if (props.onSetLoading) {
    props.onSetLoading(true);
  } else {
    emit('set-loading', true);
  }

  try {
    const r = await fetch(host + 'api/setup/' + props.meterid + '/finish', {
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
      router.push({ path: '/meter/' + props.meterid });
    } else {
      console.log('Error finishing setup');
    }
  } catch (e) {
    console.error('finishSetup failed', e);
  } finally {
    if (props.onSetLoading) {
      props.onSetLoading(false);
    } else {
      emit('set-loading', false);
    }
  }
};

const finishSetup = async () => {

  // check if initial value is not 0
  if (initialValue.value === 0) {
    dialog.warning({
      title: 'Initial value',
      content: 'Please enter a valid initial value'
    });
    return;
  }

  if (!props.confidenceThreshold) {
    dialog.warning({
      title: 'No confidence threshold set',
      content: 'Without a confidence threshold, uncertain digit predictions will be accepted as-is, which can lead to misreadings. It is strongly recommended to set a threshold.\n\nContinue without a threshold?',
      positiveText: 'Continue anyway',
      negativeText: 'Go back',
      onPositiveClick: doFinishSetup,
    });
    return;
  }

  await doFinishSetup();
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
  width: 18px;
  height: auto;
}

.digit_small{
  margin: 0px;
  width: 16px;
  height: auto;
}

.prediction{
  font-size: 30px;
}

.prediction_small{
  margin-top: -5px;
  font-size: 20px;
  cursor: help;
}

.confidence{
  font-size: 10px;
}

.grid-container{
  text-align: center;
  line-height: 0.95;
}
.th {
  border: 1px solid rgba(255, 255, 255, 0.16);
}
.card-title{
  text-transform: uppercase;
  width:100%;
  background-color: rgba(125, 125, 125, 0.1);
  text-align: center;
}

</style>