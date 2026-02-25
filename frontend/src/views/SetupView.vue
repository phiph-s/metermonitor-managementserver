<template>

  <n-h2>Setup for {{ id }}</n-h2>
  <n-steps :current="currentlyFocusedStep" :vertical="narrowScreen">
    <n-step
      title="Segmentation"
      style="max-width: 500px; cursor: pointer;"
      @click="() => {if (currentlyFocusedStep !== 1 && currentStep > 0) {currentlyFocusedStep = 1;}}"
    >
      <div v-if="(narrowScreen && currentlyFocusedStep === 1) || (!narrowScreen)">
        <div @click.stop>
        <SegmentationConfigurator
            :last-picture="lastPicture"
            :extended-last-digit="extendedLastDigit"
            :last-3-digits-narrow="last3DigitsNarrow"
            :segments="segments"
            :rotated180="rotated180"
            :roi-extractor="roiExtractor"
            :segment-mode="segmentMode"
            :capturing="capturing"
            :template-points="templatePoints"
            :digit-quads="templateDigitQuads"
            :template-ready="templateReady"
            :template-saving="templateSaving"
            :evaluation="evaluation"
            :timestamp="lastPicture ? lastPicture.picture.timestamp : ''"
            :loading="loading"
            :no-bounding-box="noBoundingBox"
            :reevaluate-error="reevaluateError"
            @update="(newSettings) => setupStore.updateSegmentationSettings(newSettings, id)"
            @next="onSegmentationNext"
            @recapture="() => setupStore.triggerCapture(id)"
            @update-template-points="(points) => { templatePoints.value = points }"
            @update-digit-quads="(quads) => { templateDigitQuads.value = quads }"
            @save-template="() => setupStore.saveTemplate(id, templatePoints.value, templateDigitQuads.value)"
        />
        </div>
        <br>
        <n-tooltip trigger="hover" placement="bottom">
          <template #trigger>
            <n-button circle type="info" ghost>
              <template #icon>
                <n-icon><HelpOutlineFilled /></n-icon>
              </template>
            </n-button>
          </template>
          <div style="max-width: 300px">
            <img src="@/assets/example_segmentation.png" alt="Example segmentation" style="max-width: 100%"/>
            <ul>
              <li>Numbers should be entirely visible</li>
              <li>One number per segement</li>
            </ul>
          </div>
        </n-tooltip>
      </div>
    </n-step>
    <n-step
      title="Threshold Extraction"
      style="max-width: 500px; cursor: pointer;"
      @click="() => {if (currentlyFocusedStep !== 2  && currentStep > 1) {currentlyFocusedStep = 2;}}"
    >
      <div v-if="(narrowScreen && currentlyFocusedStep === 2) || (!narrowScreen && currentStep > 1)">
        <div @click.stop>
          <ThresholdPicker
              :evaluation="evaluation"
              :run="tresholdedImages[tresholdedImages.length-1]"
              :threshold="threshold"
              :threshold_last="thresholdLast"
              :islanding_padding="islandingPadding"
              :segments="segments"
              :loading="loading"
              :searching-thresholds="searchingThresholds"
              :threshold-search-result="thresholdSearchResult"
              @update="(data) => setupStore.updateThresholds(data, id)"
              @reevaluate="() => setupStore.redoDigitEval(id) && setupStore.clearEvaluationExamples(id)"
              @next="() =>  {setupStore.nextStep(2); currentlyFocusedStep = 3}"
              @search-thresholds="(steps) => setupStore.searchThresholds(id, steps)"
          />
        </div>
        <br>
        <n-tooltip trigger="hover" placement="bottom">
          <template #trigger>
            <n-button circle type="info" ghost>
              <template #icon>
                <n-icon><HelpOutlineFilled /></n-icon>
              </template>
            </n-button>
          </template>
          <div style="max-width: 300px">
            <img src="@/assets/example_thresholds.png" alt="Example segmentation" style="max-width: 100%"/>
            <ul>
              <li>Select thresholds to extract numbers</li>
              <li>Numbers should be clearly visible</li>
              <li>Use the "evaluate" button to test the values</li>
              <li>Use "extraction padding" to remove as much artifacts as possible</li>
            </ul>
          </div>
        </n-tooltip>
      </div>
    </n-step>
    <n-step
      title="Evaluation Preview"
      v-if="lastPicture"
      style="max-width: 500px; cursor: pointer;"
      @click="() => {if (currentlyFocusedStep !== 3 && currentStep > 2) {currentlyFocusedStep = 3;}}"
    >
      <div v-if="(narrowScreen && currentlyFocusedStep === 3) || (!narrowScreen && currentStep > 2)">
        <div @click.stop>
          <EvaluationConfigurator
              :evaluation="evaluation"
              :max-flow-rate="maxFlowRate"
              :use-correction-alg="useCorrectionAlg"
              :loading="loading"
              @update-max-flow="(data) => setupStore.updateMaxFlow(data, id)"
              @update-conf-threshold="(data) => setupStore.updateConfThreshold(data, id)"
              @update-use-correction="(data) => setupStore.updateUseCorrection(data, id)"
              @set-loading="setupStore.setLoading"
              @request-random-example="() => setupStore.requestReevaluatedDigits(id)"
              @clear-evaluations="() => setupStore.clearEvaluationExamples()"
              :meterid="id"
              :confidence-threshold="confThreshold"
              :timestamp="lastPicture.picture.timestamp"
              :randomExamples="randomExamples"
          />
        </div>
         <br>
        <n-tooltip trigger="hover" placement="bottom">
          <template #trigger>
            <n-button circle type="info" ghost>
              <template #icon>
                <n-icon><HelpOutlineFilled /></n-icon>
              </template>
            </n-button>
          </template>
          <div style="max-width: 300px">
            <ul>
              <li>Check the values the model extracted</li>
              <li>Reflections and uneven lighting can cause issues</li>
              <li>Manually enter the correct value without a decimal point or leading zeros</li>
            </ul>
          </div>
        </n-tooltip>
      </div>
    </n-step>
  </n-steps>

</template>

<script setup>
import {onMounted, onUnmounted, computed, ref, watch} from 'vue';
import { NSteps, NStep, NButton, NH2, NTooltip, NIcon } from 'naive-ui';
import { HelpOutlineFilled } from '@vicons/material';
import { useRoute } from 'vue-router';
import { storeToRefs } from 'pinia';
import { useWatermeterStore } from '@/stores/watermeterStore';
import { useSetupStore } from '@/stores/setupStore';
import { useHeaderControls } from '@/composables/headerControls';
import SegmentationConfigurator from "@/components/SegmentationConfigurator.vue";
import ThresholdPicker from "@/components/ThresholdPicker.vue";
import EvaluationConfigurator from "@/components/EvaluationConfigurator.vue";

const route = useRoute();
const id = route.params.id;

// Initialize stores
const watermeterStore = useWatermeterStore();
const setupStore = useSetupStore();
const headerControls = useHeaderControls();

const narrowScreen = computed(() => window.innerWidth < 1500);

const currentlyFocusedStep = ref(1);

// Get reactive state from stores
const { lastPicture, evaluation, settings } = storeToRefs(watermeterStore);
const { currentStep, randomExamples, noBoundingBox, loading, searchingThresholds, thresholdSearchResult, capturing, templateSaving, templateData, reevaluateError } = storeToRefs(setupStore);

const threshold = computed(() => [settings.value?.threshold_low || 0, settings.value?.threshold_high || 0]);
const thresholdLast = computed(() => [settings.value?.threshold_last_low || 0, settings.value?.threshold_last_high || 0]);
const islandingPadding = computed(() => settings.value?.islanding_padding || 0);
const segments = computed(() => settings.value?.segments || 0);
const extendedLastDigit = computed(() => settings.value?.extended_last_digit || false);
const last3DigitsNarrow = computed(() => settings.value?.shrink_last_3 || false);
const rotated180 = computed(() => settings.value?.rotated_180 || false);
const roiExtractor = computed(() => settings.value?.roi_extractor || 'yolo');
const segmentMode = computed(() => settings.value?.segment_mode || 'display');
const templateId = computed(() => settings.value?.template_id || null);
const templateReady = computed(() => !!templateId.value);
const isTemplateExtractor = (value) => ['orb', 'static_rect'].includes(value);
const maxFlowRate = computed(() => settings.value?.max_flow_rate || 0);
const confThreshold = computed(() => settings.value?.conf_threshold);
const useCorrectionAlg = computed(() => settings.value?.use_correctional_alg ?? true);

const templatePoints = ref([]);
const templateDigitQuads = ref([]);
const pendingTemplateSave = ref(false);

watch(templateId, async (next) => {
  if (!next) return;
  await setupStore.fetchTemplate(next);
  const template = templateData.value;
  if (!template?.config?.display_corners || !template.image_width || !template.image_height) return;
  templatePoints.value = template.config.display_corners.map((point) => ({
    x: point[0] / template.image_width,
    y: point[1] / template.image_height
  }));
  if (template?.config?.digit_corners?.length) {
    templateDigitQuads.value = template.config.digit_corners.map((quad) =>
      quad.map((point) => ({
        x: point[0] / template.image_width,
        y: point[1] / template.image_height
      }))
    );
  } else {
    templateDigitQuads.value = [];
  }
});

watch(
  () => [roiExtractor.value, segmentMode.value],
  ([nextExtractor, nextMode], [prevExtractor, prevMode]) => {
    if (!isTemplateExtractor(nextExtractor)) {
      pendingTemplateSave.value = false;
      return;
    }
    if (nextExtractor !== prevExtractor || nextMode !== prevMode) {
      pendingTemplateSave.value = true;
    }
  }
);

watch(
  () => [pendingTemplateSave.value, templateId.value, templateSaving.value, templatePoints.value, templateDigitQuads.value, lastPicture.value, segmentMode.value, segments.value],
  ([pending, nextTemplateId, saving, points, digitQuads, picture, mode, segmentCount]) => {
    if (!pending || saving || nextTemplateId) return;
    if (!picture?.picture?.data) return;
    if (mode === 'display' && (!points || points.length !== 4)) return;
    if (mode === 'each_digit' && (!digitQuads || digitQuads.length !== segmentCount)) return;
    pendingTemplateSave.value = false;
    setupStore.saveTemplate(id, points, digitQuads);
  },
  { deep: true }
);

// Computed property for tresholdedImages (keeping backwards compatibility)
const tresholdedImages = computed(() => {
  return evaluation.value?.tresholded_images || [];
});

// Handler for segmentation step completion - triggers auto threshold search
const onSegmentationNext = () => {
  setupStore.nextStep(1);
  currentlyFocusedStep.value = 2;

  // Automatically start threshold search with default depth (10)
  if (import.meta.env.VITE_E2E_SKIP_THRESHOLD_SEARCH !== 'true') {
    setupStore.searchThresholds(id, 10);
  }
};

onMounted(() => {
  setupStore.reset();
  setupStore.getData(id);
  if (headerControls) {
    headerControls.setHeader({
      showRefresh: true,
      onRefresh: () => setupStore.getData(id),
      refreshLoading: loading.value
    });
  }
});

onUnmounted(() => {
  if (headerControls) {
    headerControls.resetHeader();
  }
});

watch(loading, (next) => {
  if (!headerControls) return;
  headerControls.setHeader({ refreshLoading: next });
});
</script>

<style scoped>
</style>
