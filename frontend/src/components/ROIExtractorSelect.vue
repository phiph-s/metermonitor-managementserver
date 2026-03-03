<template>
  <div class="roi-row">
    <div class="roi-label">
      <n-tooltip>
        <template #trigger>
          <span>
            <n-icon size="20"><CropOutlined /></n-icon>
          </span>
        </template>
        <span>Select the region-of-interest extractor</span>
      </n-tooltip>
    </div>
    <div class="roi-select-wrap">
      <n-select
        class="roi-select"
        :consistent-menu-width="false"
        :value="value"
        :options="options"
        :disabled="disabled"
        @update:value="(next) => emit('update:value', next)"
      />
    </div>
    <n-tooltip v-if="templateEnabled" class="roi-action">
      <template #trigger>
        <n-button
          size="small"
          :type="statusType"
          :loading="templateSaving"
          :disabled="disabled || templateSaving || !canSaveTemplate"
          @click="emit('save-template')"
        >
          <template #icon>
            <n-icon>
              <SaveOutlined />
            </n-icon>
          </template>
          Apply
        </n-button>
      </template>
      <span v-if="status === 'saved'">Template saved</span>
      <span v-else>Unapplied changes — save template</span>
    </n-tooltip>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { NSelect, NTooltip, NIcon, NButton } from 'naive-ui';
import { CropOutlined, SaveOutlined } from '@vicons/material';

const props = defineProps({
  value: { type: String, required: true },
  options: { type: Array, required: true },
  disabled: { type: Boolean, default: false },
  templateEnabled: { type: Boolean, default: false },
  templateSaving: { type: Boolean, default: false },
  templateReady: { type: Boolean, default: false },
  templateDirty: { type: Boolean, default: false },
  canSaveTemplate: { type: Boolean, default: false }
});

const emit = defineEmits(['update:value', 'save-template']);

const status = computed(() => (props.templateReady && !props.templateDirty ? 'saved' : 'dirty'));
const statusType = computed(() => (status.value === 'saved' ? 'success' : 'warning'));
</script>

<style scoped>
.tooltip-trigger {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.roi-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.roi-label {
  flex: 0 0 auto;
}

.roi-select-wrap {
  min-width: 0;
}

.roi-select {
  width: 100%;
}

.roi-select :deep(.n-base-selection) {
  width: 100%;
  min-width: 0 !important;
  max-width: 100%;
}

.roi-select :deep(.n-base-selection-input) {
  overflow: hidden;
}

.roi-select :deep(.n-base-selection-label) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}
</style>
