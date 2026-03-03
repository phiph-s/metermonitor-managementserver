<template>
  <n-card size="small">
    <template #cover>
      <div class="card-title">
        <span style="font-weight: bolder;">Base Settings</span><span style="opacity: 0.7"> - change immediately</span>
      </div>
    </template>

    <n-flex justify="space-between" align="center" style="padding-top: 16px;">
     <n-tooltip>
        <template #trigger>
          Number of Digits
        </template>
        <span>Number of segments (2-10)</span>
      </n-tooltip>
      <n-input-number
        data-testid="segments-input"
        :value="segments"
        @update:value="handleUpdate('segments', $event)"
        :max="10"
        :min="2"
        :disabled="loading">
      </n-input-number>
    </n-flex>

  </n-card><br>
  <n-card size="small">
    <template #cover>
      <div class="card-title">
        <span style="font-weight: bolder;">Extract</span><span style="opacity: 0.7"> all digits</span>
      </div>
      <div class="image-container">
        <TemplatePointEditor
          v-if="isTemplateExtractor && lastPicture && segmentModeValue === 'display'"
          :image-src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data"
          :points="templatePoints"
          @update:points="emits('updateTemplatePoints', $event)"
        />
        <TemplateDigitEditor
          v-else-if="isTemplateExtractor && lastPicture && segmentModeValue === 'each_digit'"
          :image-src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data"
          :quads="digitQuads"
          :segments="segments"
          :display-corners="templatePoints"
          @update:quads="emits('updateDigitQuads', $event)"
        />
        <img v-else-if="lastPicture && lastPicture.picture.data_bbox" :src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data_bbox" alt="Watermeter" />
        <img v-else-if="lastPicture" :src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data" alt="Watermeter" />

        <span class="timestamp-overlay">{{ formattedTimestamp }}</span>
        <n-button
          size="tiny"
          type="primary"
          class="recapture-button"
          :loading="capturing"
          :disabled="loading || capturing"
          @click="emits('recapture')"
        >
          <template #icon>
            <n-icon><CameraAltOutlined /></n-icon>
          </template>
          Recapture
        </n-button>
        <div v-if="capturing" class="capture-overlay">
          <n-spin size="small" />
        </div>
      </div>
    </template>
    <div style="padding-top: 16px;"></div>
    <ROIExtractorSelect
      :value="currentExtractor"
      :options="extractorOptions"
      :disabled="loading"
      :template-enabled="isTemplateExtractor"
      :template-saving="templateSaving"
      :template-ready="templateReady"
      :can-save-template="!!lastPicture"
      @update:value="handleUpdate('roiExtractor', $event)"
      @save-template="emits('saveTemplate')"
    />
    <template v-if="isTemplateExtractor" >
      <br>
      <n-flex align="center" :size="8" class="padd">
        <n-switch
          :value="segmentModeValue"
          :checked-value="'each_digit'"
          :unchecked-value="'display'"
          @update:value="handleUpdate('segmentMode', $event)"
          :disabled="loading"
        >
          <template #icon>
            <n-icon size="16"><GridViewOutlined /></n-icon>
          </template>
        </n-switch>
        <n-tooltip>
          <template #trigger>
            <span class="tooltip-trigger">
              <span>Segmentation mode: <b>{{ segmentModeValue === 'each_digit' ? 'Each digit' : 'Display (auto)' }}</b></span>
            </span>
          </template>
          <span>Display: select one large region.<br>Each digit: select one quad per digit.</span>
        </n-tooltip>
      </n-flex>
    </template>
    <div v-if="!isEachDigitMode">
      <br>
      <n-flex align="center" :size="8" class="padd">
        <n-switch
          :value="extendedLastDigit"
          @update:value="handleUpdate('extendedLastDigit', $event)"
          :disabled="loading || isEachDigitMode"
        >
          <template #icon>
            <n-icon size="16"><AddCircleOutlineOutlined /></n-icon>
          </template>
        </n-switch>
        <n-tooltip>
          <template #trigger>
            <span class="tooltip-trigger">
              <span>Extended last digit</span>
            </span>
          </template>
          <span>Enable if the last digits display is bigger<br>compared to the other digits</span>
        </n-tooltip>
      </n-flex>
      <n-flex align="center" :size="8" class="padd">
        <n-switch
          :value="last3DigitsNarrow"
          @update:value="handleUpdate('last3DigitsNarrow', $event)"
          :disabled="loading || isEachDigitMode"
        >
          <template #icon>
            <n-icon size="16"><CompressOutlined /></n-icon>
          </template>
        </n-switch>
        <n-tooltip>
          <template #trigger>
            <span class="tooltip-trigger">
              <span>Last 3 digits are narrow</span>
            </span>
          </template>
          <span>Enable if the last three digits displays are narrower<br>compared to the other digits</span>
        </n-tooltip>
      </n-flex>
    </div>
  </n-card><br>
  <n-card size="small">
    <template #cover>
      <div class="card-title">
        <span style="font-weight: bolder;">Post-processing</span><span style="opacity: 0.7"> & Preview</span>
      </div>
    </template>
    <div style="padding-top: 16px;"></div>
    <n-alert v-if="noBoundingBox && !isTemplateExtractor" title="No bounding box found" type="warning" style="margin-bottom: 15px;">
      Without a bounding box the segmentation will not work. Adjust the camera angle or lighting and try again.
    </n-alert>
    <n-alert v-if="reevaluateError" title="Extraction failed" type="warning" style="margin-bottom: 15px;">
      {{ reevaluateError }}
    </n-alert>

    <n-flex align="center" :size="8">
      <n-switch
        :value="rotated180"
        @update:value="handleUpdate('rotated180', $event)"
        :disabled="loading"
      >
        <template #icon>
          <n-icon size="16"><RotateRightOutlined /></n-icon>
        </template>
      </n-switch>
      <n-tooltip>
        <template #trigger>
          <span class="tooltip-trigger">
            <span>180° rotated</span>
          </span>
        </template>
        <span>Enable if the captured image is rotated 180°</span>
      </n-tooltip>
    </n-flex>
    <template #action v-if="evaluation">
      <n-flex justify="space-around" :size="[0,0]" v-if="evaluation['colored_digits']">
        <div v-for="[index, base64] in evaluation['colored_digits'].entries()" :key="base64" style="text-align: center;">
          <img :style="`width:calc(350px / ${evaluation['colored_digits'].length});`" class="digit"  :src="'data:image/png;base64,' + base64" alt="D"/><br>
          <n-flex justify="center" style="position: relative; top: -10px;">
            <div class="digit-handle">
              {{index + 1}}
            </div>
          </n-flex>
        </div>
      </n-flex><br>
      <n-flex justify="end">
        <n-button
            @click="emits('next')"
            round
            :disabled="loading || (isTemplateExtractor && !templateReady)"
            :loading="loading"
        >Next</n-button>
      </n-flex>
    </template>
  </n-card>
</template>

<script setup>
import {NCard, NFlex, NInputNumber, NSwitch, NButton, NTooltip, NAlert, NSpin, NIcon} from 'naive-ui';
import {defineProps, defineEmits, computed} from 'vue';
import {
  AddCircleOutlineOutlined,
  CameraAltOutlined,
  CompressOutlined,
  RotateRightOutlined,
  GridViewOutlined
} from '@vicons/material';
import TemplatePointEditor from '@/components/TemplatePointEditor.vue';
import TemplateDigitEditor from '@/components/TemplateDigitEditor.vue';
import ROIExtractorSelect from '@/components/ROIExtractorSelect.vue';

const props = defineProps([
    'lastPicture',
    'segments',
    'timestamp',
    'extendedLastDigit',
    'last3DigitsNarrow',
    'evaluation',
    'rotated180',
    'roiExtractor',
    'segmentMode',
    'templatePoints',
    'digitQuads',
    'templateReady',
    'templateSaving',
    'capturing',
    'loading',
    'noBoundingBox',
    'reevaluateError'
]);
const emits = defineEmits(['update', 'next', 'recapture', 'updateTemplatePoints', 'updateDigitQuads', 'saveTemplate']);

const formattedTimestamp = computed(() => {
  if (!props.timestamp) return '';
  const date = new Date(props.timestamp);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' · ' + date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
});

const extractorOptions = [
  { label: 'AUTO - Use the YOLOv11 AI-model', value: 'yolo' },
  { label: 'BYPASS - Use whole image', value: 'bypass' },
  { label: 'ORB - Feature-aligning extractor', value: 'orb' },
  { label: 'STATIC RECT - Fixed rectangle', value: 'static_rect' }
];

const currentExtractor = computed(() => props.roiExtractor || 'yolo');
const isTemplateExtractor = computed(() => ['orb', 'static_rect'].includes(currentExtractor.value));
const segmentModeValue = computed(() => props.segmentMode || 'display');
const isEachDigitMode = computed(() => segmentModeValue.value === 'each_digit');

const handleUpdate = (field, value) => {
  emits('update', {
    segments: field === 'segments' ? value : props.segments,
    extendedLastDigit: field === 'extendedLastDigit' ? value : props.extendedLastDigit,
    last3DigitsNarrow: field === 'last3DigitsNarrow' ? value : props.last3DigitsNarrow,
    rotated180: field === 'rotated180' ? value : props.rotated180,
    roiExtractor: field === 'roiExtractor' ? value : props.roiExtractor,
    segmentMode: field === 'segmentMode' ? value : props.segmentMode
  });
};

</script>


<style scoped>
.image-container {
  position: relative;
  display: inline-block;
  width: 100%;
}

.image-container img {
  width: 100%;
  display: block;
}

.timestamp-overlay {
  position: absolute;
  bottom: 8px;
  right: 8px;
  background: rgba(0, 0, 0, 0.6);
  color: rgba(255, 255, 255, 0.9);
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  backdrop-filter: blur(4px);
}

.card-title{
  text-transform: uppercase;
  width:100%;
  background-color: rgba(125, 125, 125, 0.1);
  text-align: center;
}

.recapture-button {
  position: absolute;
  bottom: 8px;
  left: 8px;
  z-index: 2;
}

.capture-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.35);
  border-radius: 4px;
}

.digit {
  height: auto;
}

.padd {
  margin-bottom: 10px;
}

.digit-handle {
  position: absolute;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: rgba(30, 30, 30, 0.8);
  border: 2px solid rgba(255, 255, 255, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: grab;
  color: white;
  z-index: 2;
}


</style>
