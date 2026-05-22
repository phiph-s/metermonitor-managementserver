<template>
  <n-modal
    v-model:show="showModel"
    preset="card"
    title="Settings"
    style="max-width: 480px; width: 92vw;"
    :mask-closable="!saving"
    :closable="!saving"
  >
    <n-space vertical size="large">

      <n-alert type="info" :show-icon="false">
        Thresholds, segments, and other segmentation parameters can only be changed in
        <b>Setup mode</b>.
      </n-alert>

      <n-form label-placement="left" label-width="180" :disabled="saving">
        <n-form-item label="Max flow rate">
          <n-input-number
            v-model:value="draft.max_flow_rate"
            :min="0"
            :step="0.1"
            :precision="2"
            style="width: 100%;"
          >
            <template #suffix>m³/h</template>
          </n-input-number>
        </n-form-item>

        <n-form-item label="Conf. threshold">
          <n-input-number
            v-model:value="draft.conf_threshold"
            :min="0"
            :max="100"
            :step="1"
            :precision="1"
            placeholder="Auto"
            clearable
            style="width: 100%;"
          >
            <template #suffix>%</template>
          </n-input-number>
        </n-form-item>

        <n-form-item label="Rotated 180°">
          <n-switch v-model:value="draft.rotated_180" />
        </n-form-item>

        <n-form-item label="Meter Type">
          <n-select
            v-model:value="draft.meter_type"
            :options="meterTypeOptions"
            style="width: 100%;"
          />
        </n-form-item>

        <n-form-item v-if="draft.meter_type === 'CUSTOM'" label="Unit">
          <n-input
            v-model:value="draft.unit"
            placeholder="e.g. kWh, m³, L"
            style="width: 100%;"
          />
        </n-form-item>

        <template v-if="!isEachDigitMode">
          <n-form-item label="Extended last digit">
            <n-switch v-model:value="draft.extended_last_digit" />
          </n-form-item>

          <n-form-item label="Last 3 digits narrow">
            <n-switch v-model:value="draft.shrink_last_3" />
          </n-form-item>
        </template>
      </n-form>

      <n-divider style="margin: 0;" />

      <div>
        <div class="section-title">Set read value</div>
        <div class="section-desc">
          Manually set the current meter reading to get the correction algorithm back on track.
          This adds a manual entry to the history and marks all evaluations as outdated.
        </div>
        <n-flex style="margin-top: 10px;" align="center">
          <n-input-number
            v-model:value="readValue"
            :min="0"
            placeholder="e.g. 43300"
            :disabled="settingValue"
            style="flex: 1; min-width: 140px;"
          />
          <n-button
            type="primary"
            tertiary
            :disabled="readValue === null"
            :loading="settingValue"
            @click="doSetReadValue"
          >
            Apply
          </n-button>
        </n-flex>
        <div v-if="readValueError" class="msg error">{{ readValueError }}</div>
        <div v-if="readValueSuccess" class="msg success">Read value set successfully.</div>
      </div>

    </n-space>

    <template #action>
      <n-space justify="end">
        <n-button :disabled="saving" @click="showModel = false">Cancel</n-button>
        <n-button type="primary" :loading="saving" @click="save">Save settings</n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<script setup>
import { computed, reactive, ref, watch, h } from 'vue';
import {
  NModal, NSpace, NFlex, NForm, NFormItem,
  NInputNumber, NSwitch, NButton, NAlert, NDivider, NSelect, NInput,
  useMessage
} from 'naive-ui';
import { meterTypeColors, meterTypeLabels, METER_TYPES } from '@/utils/meterTypeMeta';
import { useWatermeterStore } from '@/stores/watermeterStore';
import { apiService } from '@/services/api';

const props = defineProps({
  show: Boolean,
  meterId: String,
});
const emit = defineEmits(['update:show', 'saved']);

const store = useWatermeterStore();
const message = useMessage();

const showModel = computed({
  get: () => props.show,
  set: (val) => emit('update:show', val),
});

const isEachDigitMode = computed(() => store.settings?.segment_mode === 'each_digit');

const meterTypeOptions = METER_TYPES.map(t => ({
  label: meterTypeLabels[t],
  value: t,
  renderLabel: () => h('span', [
    h('span', { style: { color: meterTypeColors[t], fontWeight: 700, marginRight: '6px' } }, '●'),
    meterTypeLabels[t],
  ]),
}));

// --- Settings draft ---
const draft = reactive({
  max_flow_rate: null,
  conf_threshold: null,
  rotated_180: false,
  extended_last_digit: false,
  shrink_last_3: false,
  meter_type: 'WATER',
  unit: null,
});

watch(() => props.show, (val) => {
  if (val) {
    draft.max_flow_rate = store.settings.max_flow_rate ?? null;
    draft.conf_threshold = store.settings.conf_threshold ?? null;
    draft.rotated_180 = store.settings.rotated_180 ?? false;
    draft.extended_last_digit = store.settings.extended_last_digit ?? false;
    draft.shrink_last_3 = store.settings.shrink_last_3 ?? false;
    draft.meter_type = store.settings.meter_type || 'WATER';
    draft.unit = store.settings.unit ?? null;
    readValue.value = null;
    readValueError.value = '';
    readValueSuccess.value = false;
  }
});

const saving = ref(false);

const save = async () => {
  if (saving.value) return;
  saving.value = true;
  try {
    // Build full payload: keep all threshold/segmentation fields from the store untouched,
    // only override the fields exposed in this modal.
    const s = store.settings;
    await apiService.put(`api/watermeters/${props.meterId}/settings`, {
      threshold_low: s.threshold_low,
      threshold_high: s.threshold_high,
      threshold_last_low: s.threshold_last_low,
      threshold_last_high: s.threshold_last_high,
      islanding_padding: s.islanding_padding,
      segments: s.segments,
      roi_extractor: s.roi_extractor,
      template_id: s.template_id,
      segment_mode: s.segment_mode,
      use_correctional_alg: s.use_correctional_alg,
      digit_models: s.digit_models,
      decimals: s.decimals,
      // Fields editable in this modal:
      max_flow_rate: draft.max_flow_rate,
      conf_threshold: draft.conf_threshold,
      rotated_180: draft.rotated_180,
      extended_last_digit: draft.extended_last_digit,
      shrink_last_3: draft.shrink_last_3,
      meter_type: draft.meter_type,
      unit: draft.unit,
    });
    // Sync store to reflect saved changes (no full refetch needed for these fields)
    s.max_flow_rate = draft.max_flow_rate;
    s.conf_threshold = draft.conf_threshold;
    s.rotated_180 = draft.rotated_180;
    s.extended_last_digit = draft.extended_last_digit;
    s.shrink_last_3 = draft.shrink_last_3;
    s.meter_type = draft.meter_type;
    s.unit = draft.unit;
    message.success('Settings saved.');
    emit('saved');
    showModel.value = false;
  } catch (e) {
    message.error('Failed to save settings.');
  } finally {
    saving.value = false;
  }
};

// --- Set read value ---
const readValue = ref(null);
const settingValue = ref(false);
const readValueError = ref('');
const readValueSuccess = ref(false);

const doSetReadValue = async () => {
  if (readValue.value === null || settingValue.value) return;
  settingValue.value = true;
  readValueError.value = '';
  readValueSuccess.value = false;
  try {
    await apiService.post(
      `api/watermeters/${props.meterId}/set-read-value`,
      { value: readValue.value }
    );
    readValueSuccess.value = true;
    readValue.value = null;
    emit('saved');
  } catch (e) {
    readValueError.value = 'Failed to set read value.';
  } finally {
    settingValue.value = false;
  }
};
</script>

<style scoped>
.section-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 4px;
}
.section-desc {
  font-size: 12px;
  opacity: 0.65;
  line-height: 1.5;
}
.msg {
  margin-top: 6px;
  font-size: 12px;
}
.msg.error { color: #d03050; }
.msg.success { color: #18a058; }
</style>
