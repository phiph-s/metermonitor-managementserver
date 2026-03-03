<template>
  <template v-if="evaluation">
    <n-card size="small">
      <template #cover>
        <div class="card-title">
          <span style="font-weight: bolder;">Result preview</span><span style="opacity: 0.7"> & Benchmark</span>
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
      <n-flex align="center">
        <n-badge
            v-if="randomExamples && randomExamples.length > 0 && !useSetupStore().runningBenchmark"
          :value="loading ? '' : `${averageConfidence}%`"
          :processing="loading"
          :type="getBadgeType(averageConfidence)"
          :show="randomExamples && randomExamples.length > 0"
        >
        </n-badge>
        <span style="font-weight: 500;" v-if="randomExamples && randomExamples.length > 0 && !useSetupStore().runningBenchmark">
          Average Confidence on digits from {{ randomExamples.length }} random historical evaluation{{ randomExamples.length > 1 ? 's' : '' }}.
          <n-popconfirm @positive-click="clearHistoryAndRebenchmark">
            <template #trigger>
              <a href="#" @click.prevent style="font-weight: normal; font-size: 12px; margin-left: 8px; opacity: 0.6;">Clear history</a>
            </template>
            This will delete all historic evaluations. Are you sure?
          </n-popconfirm>
        </span>
        <template v-else-if="useSetupStore().runningBenchmark">
          Benchmarking on past evaluations...
          <n-progress type="line" :percentage="randomExamples.length * 10" :show-indicator="false" style="width: 100px;"></n-progress>
        </template>
        <span v-else-if="useSetupStore().tooFewEvaluations" style="opacity: 0.6;">
          Too few historical images to run benchmark.
        </span>
        <span v-else>
          Press "Apply" to evaluate and run the benchmark.
        </span>
      </n-flex>
      <br>
      <n-collapse v-if="randomExamples && randomExamples.length > 0">
        <n-collapse-item title="Show results" name="1">
          <n-grid :cols="evaluation['th_digits'].length * 2" y-gap="4">
            <template v-for="[i, example] in randomExamples.entries()" :key="i + 'example'">
              <n-gi justify="space-around" size="small" v-for="[i,base64] in example['processed_images'].entries()" :key="i + 'x'" class="grid-container">
                <img
                    class="digit_small theme-revert"
                    :src="'data:image/png;base64,' + base64"
                    alt="D"
                />
                <br>
                <span
                    class="prediction_small google-sans-code"
                    :style="{color: getColor(example['predictions'][i][0][1])}"
                    :title="`${example['predictions'][i][0][0]}: ${(example['predictions'][i][0][1]*100).toFixed(1)}\n${example['predictions'][i][1][0]}: ${(example['predictions'][i][1][1]*100).toFixed(1)}\n${example['predictions'][i][2][0]}: ${(example['predictions'][i][2][1]*100).toFixed(1)}`"
                >
                  {{ (example['predictions'][i][0][0]==='r')? '↕' : example['predictions'][i][0][0] }}
                </span>
              </n-gi>
            </template>
          </n-grid>
        </n-collapse-item>
      </n-collapse>
    </n-card><br>
    <n-card size="small">
      <template #cover>
        <div class="card-title">
          <span style="font-weight: bolder;">Evaluation Settings</span><span style="opacity: 0.7"></span>
        </div>
      </template>
      <n-flex style="padding-top: 16px;">
        <div style="max-width: 30%">
            <n-tooltip>
              <template #trigger>
                Conf. threshold
              </template>
              <span>
                Set a confidence threshold for accepting digit predictions.<br>
                Digits with confidence below this value will be marked as uncertain.
              </span>
            </n-tooltip>
          <n-input-number ref="confThreshInput" :value="confidenceThreshold" @update:value="(e) => {emit('update-conf-threshold', e); lastConfThresholdInput = e;}"
                          :placeholder="`${Math.max(averageConfidence-20,0)}%`" :disabled="loading" />
        </div>
        <div style="max-width: 33%">
          Read initial value
          <n-input-number v-model:value="initialValue" placeholder="Readout" :disabled="loading" />
        </div>
        <div style="max-width: 30%">
          Max. flow rate
          <n-input-number :value="maxFlowRate"
                          @update:value="emit('update-max-flow', $event)"
                          placeholder="Flow rate" :disabled="loading || !useCorrectionAlg" />
        </div>
        <div style="max-width: 30%">
          <n-tooltip>
            <template #trigger>
              Correctional alg.
            </template>
            <span>
              Full: Positive flow check, max flow rate, fallback handling<br>
              Light: Only rotation and low-confidence digits corrected
            </span>
          </n-tooltip>
          <n-switch v-model:value="useCorrectionAlg" @update:value="emit('update-use-correction', $event)" :disabled="loading">
            <template #checked>Full</template>
            <template #unchecked>Light</template>
          </n-switch>
        </div>
      </n-flex>
      <template #action>
        <n-flex justify="end" size="large">
          <n-button
              @click="finishSetup"
              type="success"
              round
              :disabled="loading || useSetupStore().runningBenchmark"
              :loading="loading || useSetupStore().runningBenchmark"
          >Finish & save</n-button>
        </n-flex>
      </template>
    </n-card>
  </template>
</template>

<script setup>
import {defineProps, ref, defineEmits, computed} from 'vue';
import {
  NFlex,
  NCard,
  NButton,
  NInputNumber,
  NProgress,
  NIcon,
  NGrid,
  NGi,
  NCollapse,
  NCollapseItem,
  NBadge,
  NPopconfirm,
  useDialog,
  NTooltip,
  NTag,
  NSwitch
} from 'naive-ui';
import router from "@/router";
import {useSetupStore} from "@/stores/setupStore";
import {apiService} from "@/services/api";

const emit = defineEmits(['set-loading', 'request-random-example', 'update-max-flow', 'update-conf-threshold', 'update-use-correction', 'clear-evaluations']);

const props = defineProps([
    'meterid',
    'evaluation',
    'timestamp',
    'maxFlowRate',
    'confidenceThreshold',
    'useCorrectionAlg',
    'loading',
    'onSetLoading',
    'randomExamples'
]);

const useCorrectionAlg = ref(props.useCorrectionAlg ?? true);

const initialValue = ref(props.evaluation['predictions'].reduce((acc, digit) => {
  const predictedDigit = digit[0][0];
  return acc * 10 + (predictedDigit === 'r' ? 0 : parseInt(predictedDigit));
}, 0));

const dialog = useDialog();
const host = import.meta.env.VITE_HOST;

const confThreshInput = ref(null);
const lastConfThresholdInput = ref(props.confidenceThreshold);

// Calculate average confidence from random examples
const averageConfidence = computed(() => {
  if (!props.randomExamples || props.randomExamples.length === 0) {
    return 0;
  }

  let totalConfidence = 0;
  let count = 0;

  props.randomExamples.forEach(example => {
    if (example.predictions) {
      example.predictions.forEach(prediction => {
        if (prediction && prediction[0] && prediction[0][1] !== undefined) {
          totalConfidence += prediction[0][1];
          count++;
        }
      });
    }
  });

  return count > 0 ? ((totalConfidence / count) * 100).toFixed(1) : 0;
});

// Get badge type based on confidence level
const getBadgeType = (confidence) => {
  if (confidence >= 90) return 'success';
  if (confidence >= 75) return 'warning';
  return 'error';
};

const clearHistoryAndRebenchmark = async () => {
  try {
    const response = await apiService.delete(`api/watermeters/${props.meterid}/evals`);
    if (response.ok) {
      const result = await response.json();
      console.log(`Cleared ${result.count} evaluations`);

      // Re-evaluate latest picture first to restore setup state
      await apiService.post(`api/watermeters/${props.meterid}/evaluations/reevaluate`);

      // Emit event and re-run benchmark
      emit('clear-evaluations');
      emit('request-random-example');
    } else {
      console.error('Error clearing evaluations');
    }
  } catch (err) {
    console.error('Error clearing evaluations:', err);
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

  // notify parent to show loading
  if (props.onSetLoading) {
    props.onSetLoading(true);
  } else {
    emit('set-loading', true);
  }

  try {

    // if confThreshInput vlaue is null, set value to averageConfidence - 20
    console.log(lastConfThresholdInput.value)
    if (lastConfThresholdInput.value == null) {
      const suggestedThreshold = Math.max(averageConfidence.value - 20, 0);
      emit('update-conf-threshold', suggestedThreshold);
      console.log('Suggested confidence threshold set to', suggestedThreshold);
    }

    // post to /api/setup/{name}/finish
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
    // ensure parent loading is turned off if we didn't navigate away
    if (props.onSetLoading) {
      props.onSetLoading(false);
    } else {
      emit('set-loading', false);
    }
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