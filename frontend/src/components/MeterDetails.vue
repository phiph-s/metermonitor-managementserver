<template>
  <div style="margin: 0 auto;">
    <n-card v-if="data" :title="id" size="small">
      <template #header-extra>
        <div class="header-extra">
          <span class="timestamp">{{ formattedTimestamp }}</span>
          <n-dropdown :options="menuOptions" @select="handleMenuSelect" placement="bottom-end">
            <n-button text size="small" class="menu-trigger">
              <template #icon>
                <n-icon><MoreVertFilled /></n-icon>
              </template>
            </n-button>
          </n-dropdown>
        </div>
      </template>
      <template #cover>
        <div class="image-wrap">
          <img
            v-if="showBbox && data.picture?.data_bbox"
            :src="'data:image/' + data.picture.format + ';base64,' + data.picture.data_bbox"
            alt="Watermeter"
          />
          <img
            v-else
            :src="'data:image/' + data.picture?.format + ';base64,' + data.picture?.data"
            alt="Watermeter"
          />
          <div v-if="store.capturing" class="capture-indicator">
            <n-spin size="small" />
            <span>Capturing…</span>
          </div>
          <n-button
            v-if="data.picture?.data_bbox"
            class="bbox-toggle"
            size="tiny"
            tertiary
            @click.stop="showBbox = !showBbox"
          >
            <template #icon>
              <n-icon>
                <CropOutlined />
              </n-icon>
            </template>
            {{ showBbox ? 'BBox' : 'Raw' }}
          </n-button>
        </div>
      </template>
      <SourceCollapse
        v-if="data"
        :source="sourceForView"
        @updated="refreshSource"
      />
    </n-card>

    <div v-if="!data?.picture?.data_bbox" class="notice warning">
      <div class="notice-title">
        <n-icon><WarningAmberOutlined /></n-icon>
        <span>No Bounding Box</span>
      </div>
      <div class="notice-body">
        In the last capture, no bounding box was detected. This may indicate that the meter is not properly set up or that the image quality is poor.
      </div>
    </div>

    <div v-if="store.source?.last_error" class="notice error">
      <div class="notice-title">
        <n-icon><ErrorOutlineOutlined /></n-icon>
        <span>Source Error</span>
      </div>
      <div class="notice-body">
        {{ store.source.last_error }}
      </div>
    </div>

    <template v-if="data && data.dataset_present">
      <div class="dataset-row">
        <div class="dataset-title">
          <n-icon><ArchiveOutlined /></n-icon>
          <span>Dataset present.</span>
        </div>
        <n-space size="small" align="center">
          <n-button
            type="primary"
            tertiary
            size="small"
            :loading="downloadingDataset"
            @click="emit('downloadDataset')"
          >
            <template #icon>
              <n-icon><DownloadOutlined /></n-icon>
            </template>
            Download
          </n-button>
          <n-popconfirm @positive-click="emit('deleteDataset')">
            <template #trigger>
              <n-button type="error" tertiary size="small">
                <template #icon>
                  <n-icon><DeleteForeverFilled /></n-icon>
                </template>
                Delete
              </n-button>
            </template>
            This will clear the dataset for this meter. Are you sure?
          </n-popconfirm>
        </n-space>
      </div>
    </template>

    <MeterCharts :history="history" :daily-history="dailyHistory" :unit="meterUnit" :daily-avg="dailyAvgValue" style="margin-top: 12px;" />
    
    <template v-if="data && data['WiFi-RSSI']">
      <WifiStatus v-if="data && data['WiFi-RSSI']" :rssi="data['WiFi-RSSI']" />
      <br><br>
    </template>

    <div class="settings-title">
      <span>Settings</span>
      <n-button text size="small" @click="settingsModalOpen = true">
        <template #icon><n-icon><EditOutlined /></n-icon></template>
        Edit
      </n-button>
    </div>
    <div class="settings-grid">
      <div class="setting-chip">
        <n-icon><TuneOutlined /></n-icon>
        <div>
          <div class="chip-label">1. Thresholds</div>
          <div class="chip-value">{{ settings.threshold_low }}–{{ settings.threshold_high }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><FilterAltOutlined /></n-icon>
        <div>
          <div class="chip-label">Last Thresholds</div>
          <div class="chip-value">{{ settings.threshold_last_low }}–{{ settings.threshold_last_high }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><CropOutlined /></n-icon>
        <div>
          <div class="chip-label">Islanding</div>
          <div class="chip-value">{{ settings.islanding_padding }}%</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><ViewWeekOutlined /></n-icon>
        <div>
          <div class="chip-label">Segments</div>
          <div class="chip-value">{{ settings.segments }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><AddCircleOutlineOutlined /></n-icon>
        <div>
          <div class="chip-label">Extended last digit</div>
          <div class="chip-value">{{ settings.extended_last_digit ? 'Yes' : 'No' }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><CompressOutlined /></n-icon>
        <div>
          <div class="chip-label">last 3 digits narrow</div>
          <div class="chip-value">{{ settings.shrink_last_3 ? 'Yes' : 'No' }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><RotateRightOutlined /></n-icon>
        <div>
          <div class="chip-label">Rotated 180</div>
          <div class="chip-value">{{ settings.rotated_180 ? 'Yes' : 'No' }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><SpeedOutlined /></n-icon>
        <div>
          <div class="chip-label">Max flow</div>
          <div class="chip-value">{{ settings.max_flow_rate }} m³/h</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><CheckCircleOutlineOutlined /></n-icon>
        <div>
          <div class="chip-label">Conf. threshold</div>
          <div class="chip-value">{{ settings.conf_threshold?.toFixed(2) }}%</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><CropOutlined /></n-icon>
        <div>
          <div class="chip-label">ROI Extractor</div>
          <div class="chip-value">{{ extractorLabel }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><CheckCircleOutlineOutlined /></n-icon>
        <div>
          <div class="chip-label">Correction Alg.</div>
          <div class="chip-value">{{ settings.use_correctional_alg ? 'Full' : 'Light' }}</div>
        </div>
      </div>
      <div class="setting-chip">
        <n-icon><WaterDropOutlined /></n-icon>
        <div>
          <div class="chip-label">Meter Type</div>
          <div class="chip-value" :style="{ color: meterTypeColor }">{{ meterTypeLabel }}</div>
        </div>
      </div>
      <div v-if="settings.meter_type === 'CUSTOM'" class="setting-chip">
        <n-icon><TagOutlined /></n-icon>
        <div>
          <div class="chip-label">Unit</div>
          <div class="chip-value">{{ settings.unit || '—' }}</div>
        </div>
      </div>
    </div>
  </div>

  <MeterSettingsModal
    v-model:show="settingsModalOpen"
    :meter-id="id"
    @saved="store.fetchAll(id)"
  />

  <n-modal v-model:show="editRoiOpen" preset="card" title="Edit ROI" style="width: 720px;">
    <div v-if="editRoiLoading" class="modal-loading">
      <n-spin size="small" />
      <span>Loading template…</span>
    </div>
    <div v-else class="modal-body">
      <TemplatePointEditor
        v-if="data?.picture?.data"
        :image-src="'data:image/' + data.picture.format + ';base64,' + data.picture.data"
        :points="roiPoints"
        @update:points="(points) => { roiPoints = points }"
      />
      <div v-else class="modal-note">No reference image available.</div>
    </div>
    <div v-if="editRoiError" class="modal-error">{{ editRoiError }}</div>
    <template #action>
      <n-space justify="end">
        <n-button @click="editRoiOpen = false">Cancel</n-button>
        <n-button type="primary" :loading="editRoiSaving" @click="saveRoiTemplate">Save</n-button>
      </n-space>
    </template>
  </n-modal>
</template>

<script setup>
import { defineProps, defineEmits, computed, h, ref, onMounted, watch } from 'vue';
import {
  NCard,
  NFlex,
  NButton,
  NList,
  NPopconfirm,
  NIcon,
  NDropdown,
  NSpace,
  useDialog,
  NSpin,
  NGi,
  NModal
} from "naive-ui";
import {
  DeleteForeverFilled,
  MoreVertFilled,
  CameraAltOutlined,
  SettingsOutlined,
  DeleteOutlineOutlined,
  PlaylistRemoveOutlined,
  CropOutlined,
  TuneOutlined,
  FilterAltOutlined,
  ViewWeekOutlined,
  AddCircleOutlineOutlined,
  CompressOutlined,
  RotateRightOutlined,
  SpeedOutlined,
  CheckCircleOutlineOutlined,
  ArchiveOutlined,
  DownloadOutlined,
  WarningAmberOutlined,
  ErrorOutlineOutlined,
  RefreshOutlined,
  EditOutlined,
  WaterDropOutlined,
  TagOutlined
} from '@vicons/material';
import { getMeterUnit, getMeterTypeColor, getMeterTypeLabel } from '@/utils/meterTypeMeta';
import MeterSettingsModal from '@/components/MeterSettingsModal.vue';
import WifiStatus from "@/components/WifiStatus.vue";
import {useWatermeterStore} from "@/stores/watermeterStore";
import SourceCollapse from "@/components/SourceCollapse.vue";
import MeterCharts from "@/components/MeterCharts.vue";
import TemplatePointEditor from "@/components/TemplatePointEditor.vue";
import { apiService } from '@/services/api';

const props = defineProps({
  data: Object,
  settings: Object,
  id: String,
  downloadingDataset: Boolean,
  history: Object,
  dailyHistory: Object
});

const store = useWatermeterStore();
const dialog = useDialog();
const emit = defineEmits(['resetToSetup', 'deleteMeter', 'clearEvaluations', 'downloadDataset', 'deleteDataset', 'triggerCapture']);
const showBbox = ref(true);
const settingsModalOpen = ref(false);
const editRoiOpen = ref(false);
const dailyAvgValue = ref(null);

const fetchStats = async () => {
  if (!props.id) return;
  try {
    const s = await apiService.getJson(`api/watermeters/${props.id}/stats`);
    dailyAvgValue.value = s.daily_avg ?? null;
  } catch (_) { /* ignore */ }
};

onMounted(fetchStats);
watch(() => props.dailyHistory, fetchStats);
const editRoiLoading = ref(false);
const editRoiSaving = ref(false);
const editRoiError = ref('');
const roiPoints = ref([]);

const sourceForView = computed(() => {
  if (store.source) return store.source;
  return {
    name: props.id,
    source_type: 'mqtt',
    enabled: true
  };
});

const refreshSource = async () => {
  if (!props.id) return;
  await store.fetchSource(props.id);
};

const menuOptions = computed(() => {
  const options = [];
  if (store.source?.source_type !== 'mqtt') {
    options.push({
      label: store.capturing ? 'Capturing...' : 'Capture',
      key: 'capture',
      disabled: store.capturing,
      icon: () => h(NIcon, null, { default: () => h(CameraAltOutlined) })
    });
  }
  if (props.settings?.roi_extractor === 'orb') {
    options.push({
      label: 'Edit ROI',
      key: 'edit-roi',
      icon: () => h(NIcon, null, { default: () => h(CropOutlined) })
    });
  }
  options.push(
    { label: 'Setup', key: 'setup', icon: () => h(NIcon, null, { default: () => h(SettingsOutlined) }) },
    { label: 'Reevaluate Latest', key: 'reevaluate-latest', icon: () => h(NIcon, null, { default: () => h(RefreshOutlined) }) },
    { label: 'Clear Evals', key: 'clear-evals', icon: () => h(NIcon, null, { default: () => h(PlaylistRemoveOutlined) }) },
    { label: 'Reset Corr. Alg.', key: 'reset-corr-alg', icon: () => h(NIcon, null, { default: () => h(PlaylistRemoveOutlined) }) },
    { type: 'divider', key: 'divider' },
    { label: 'Delete', key: 'delete', icon: () => h(NIcon, { color: '#d03050' }, { default: () => h(DeleteOutlineOutlined) }) }
  );
  return options;
});

const handleMenuSelect = (key) => {
  if (key === 'edit-roi') {
    openEditRoi();
    return;
  }
  if (key === 'capture') {
    dialog.warning({
      title: 'Trigger Capture',
      content: 'This will trigger a new picture capture on the source. Are you sure?',
      positiveText: 'Capture',
      negativeText: 'Cancel',
      onPositiveClick: () => emit('triggerCapture')
    });
    return;
  }
  if (key === 'setup') {
    dialog.info({
      title: 'Enter Setup Mode',
      content: 'While the meter is in setup mode, no values will be published. Are you sure? Most settings can be changed below by pressing "Edit" without entering setup mode.',
      positiveText: 'Setup',
      negativeText: 'Cancel',
      onPositiveClick: () => emit('resetToSetup')
    });
    return;
  }
  if (key === 'clear-evals') {
    dialog.warning({
      title: 'Clear Evaluations',
      content: 'This will delete all evaluations for this meter. Are you sure?',
      positiveText: 'Clear',
      negativeText: 'Cancel',
      onPositiveClick: () => emit('clearEvaluations')
    });
    return;
  }
  if (key === 'reevaluate-latest') {
    dialog.warning({
      title: 'Reevaluate Latest',
      content: 'This will reevaluate the latest capture using current settings. Are you sure?',
      positiveText: 'Reevaluate',
      negativeText: 'Cancel',
      onPositiveClick: async () => {
        try {
          await apiService.post(`api/watermeters/${props.id}/evaluations/reevaluate?skip_setup_overwriting=true`);
          await store.fetchAll(props.id);
        } catch (e) {
          console.error('Failed to reevaluate latest:', e);
        }
      }
    });
    return;
  }
  if (key === 'reset-corr-alg') {
    dialog.warning({
      title: 'Reset Correctional Algorithm',
      content: 'This will mark all evaluations as outdated, forcing re-evaluation with the current correction settings.',
      positiveText: 'Reset',
      negativeText: 'Cancel',
      onPositiveClick: async () => {
        try {
          await apiService.post(`api/watermeters/${props.id}/evaluations/mark-outdated`);
          await store.fetchAll(props.id);
        } catch (e) {
          console.error('Failed to reset correctional algorithm:', e);
        }
      }
    });
    return;
  }
  if (key === 'delete') {
    dialog.error({
      title: 'Delete Meter',
      content: 'This will delete the meter with all its settings and data. Are you sure?',
      positiveText: 'Delete',
      negativeText: 'Cancel',
      onPositiveClick: () => emit('deleteMeter')
    });
  }
};

const openEditRoi = async () => {
  editRoiError.value = '';
  editRoiOpen.value = true;
  editRoiLoading.value = true;
  roiPoints.value = [];
  try {
    if (props.settings?.template_id) {
      const template = await apiService.getJson(`api/templates/${props.settings.template_id}`);
      if (template?.config?.display_corners?.length === 4 && template.image_width && template.image_height) {
        roiPoints.value = template.config.display_corners.map((point) => ({
          x: point[0] / template.image_width,
          y: point[1] / template.image_height
        }));
      }
    }
    if (roiPoints.value.length === 0) {
      roiPoints.value = [
        { x: 0.2, y: 0.2 },
        { x: 0.8, y: 0.2 },
        { x: 0.8, y: 0.8 },
        { x: 0.2, y: 0.8 }
      ];
    }
  } catch (e) {
    editRoiError.value = 'Failed to load template.';
  } finally {
    editRoiLoading.value = false;
  }
};

const saveRoiTemplate = async () => {
  if (editRoiSaving.value) return;
  editRoiError.value = '';
  if (!props.settings || props.settings.roi_extractor !== 'orb') {
    editRoiError.value = 'ROI extractor does not support templates.';
    return;
  }
  if (!props.data?.picture?.data) {
    editRoiError.value = 'No reference image available.';
    return;
  }
  if (!roiPoints.value || roiPoints.value.length !== 4) {
    editRoiError.value = 'Template requires 4 points.';
    return;
  }
  editRoiSaving.value = true;
  try {
    const payload = {
      name: props.id,
      extractor: props.settings.roi_extractor,
      reference_image_base64: props.data.picture.data,
      image_width: props.data.picture.width,
      image_height: props.data.picture.height,
      display_corners: roiPoints.value.map((point) => [
        Math.round(point.x * props.data.picture.width),
        Math.round(point.y * props.data.picture.height)
      ])
    };
    const result = await apiService.postJson('api/templates', payload);
    if (result?.id) {
      store.settings.roi_extractor = 'orb';
      store.settings.template_id = result.id;
      await store.updateSettings(props.id);
      const updated = await store.fetchSettings(props.id);
      if (updated?.template_id !== result.id) {
        editRoiError.value = 'Template was saved but not applied to settings.';
        return;
      }
      await apiService.post(`api/watermeters/${props.id}/evaluations/reevaluate`);
      await store.fetchAll(props.id);
      editRoiOpen.value = false;
    } else {
      editRoiError.value = 'Failed to save template.';
    }
  } catch (e) {
    editRoiError.value = 'Failed to save template.';
  } finally {
    editRoiSaving.value = false;
  }
};

const extractorLabel = computed(() => {
  const value = props.settings?.roi_extractor || 'yolo';
  if (value === 'orb') return 'ORB (Template)';
  if (value === 'bypass') return 'Bypass';
  if (value === 'yolo') return 'AUTO (YOLO)';
  return value.toUpperCase();
});

const meterUnit = computed(() => getMeterUnit(props.settings?.meter_type || 'WATER', props.settings?.unit));
const meterTypeColor = computed(() => getMeterTypeColor(props.settings?.meter_type || 'WATER'));
const meterTypeLabel = computed(() => getMeterTypeLabel(props.settings?.meter_type || 'WATER'));

const formattedTimestamp = computed(() => {
  if (!props.data?.picture?.timestamp) return '';
  const date = new Date(props.data.picture.timestamp);
  return date.toLocaleDateString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  }) + ' · ' + date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit'
  });
});
</script>

<style scoped>
.rotated {
  transform: rotate(180deg);
}

.header-extra {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.timestamp {
  line-height: 1.2;
}

.menu-trigger {
  line-height: 1;
}

.dataset-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  margin-top: 8px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
}

.light-mode .dataset-row {
  background: rgba(0, 0, 0, 0.04);
}

.dataset-title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.settings-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  margin-bottom: 6px;
  font-size: 12px;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  opacity: 0.7;
}

.notice {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
}

.light-mode .notice {
  background: rgba(0, 0, 0, 0.04);
}

.notice-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.notice-body {
  margin-top: 4px;
  font-size: 12px;
  opacity: 0.7;
}

.notice.warning {
  border-left: 4px solid #f0a900;
}

.notice.error {
  border-left: 4px solid #d03050;
}

.image-wrap {
  position: relative;
}

.capture-indicator {
  position: absolute;
  top: 8px;
  left: 8px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.6);
  font-size: 12px;
  color: #fff;
}

.light-mode .capture-indicator {
  background: rgba(255, 255, 255, 0.9);
  color: #111;
}

.bbox-toggle {
  position: absolute;
  top: 8px;
  right: 8px;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
}

.setting-chip {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.06);
}

.light-mode .setting-chip {
  background: rgba(0, 0, 0, 0.04);
}

.chip-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  opacity: 0.6;
}

.chip-value {
  font-size: 14px;
  font-weight: 600;
}

.modal-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 0;
}

.modal-body {
  padding-top: 6px;
}

.modal-note {
  font-size: 12px;
  opacity: 0.7;
}

.modal-error {
  margin-top: 8px;
  color: #d03050;
  font-size: 12px;
}
</style>
