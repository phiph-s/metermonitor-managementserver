import { defineStore } from 'pinia';
import { ref, reactive } from 'vue';
import { apiService } from '@/services/api';

export const useWatermeterStore = defineStore('watermeter', () => {
  // State
  const lastPicture = ref(null);
  const evaluations = ref([]);
  const evaluation = ref({});
  const history = ref(null);
  const source = ref(null);
  const capturing = ref(false);
  const settings = reactive({
    threshold_low: 0,
    threshold_high: 125,
    threshold_last_low: 0,
    threshold_last_high: 125,
    islanding_padding: 0,
    segments: 0,
    extended_last_digit: false,
    shrink_last_3: false,
    rotated_180: false,
    max_flow_rate: 1.0,
    conf_threshold: null,
    roi_extractor: 'yolo',
    template_id: null,
    segment_mode: 'display',
  });

  // Actions
  const fetchWatermeter = async (meterId) => {
    const data = await apiService.getJson(`api/watermeters/${meterId}`);
    lastPicture.value = data;
    return data;
  };

  const fetchEvaluations = async (meterId, amount = 20, fromId = null) => {
    let url = `api/watermeters/${meterId}/evals?amount=${amount}`;
    if (fromId) {
      url += `&from_id=${fromId}`;
    }
    const data = await apiService.getJson(url);
    if (fromId) {
      if (data.evals) {
        evaluations.value.push(...data.evals);
      }
    } else {
      evaluations.value = data.evals || [];
      if (evaluations.value.length > 0) {
        evaluation.value = evaluations.value[0];
      }
    }
    return data;
  };

  const fetchHistory = async (meterId) => {
    const data = await apiService.getJson(`api/watermeters/${meterId}/history`);
    history.value = data;
    return data;
  };

  const fetchSettings = async (meterId) => {
    const data = await apiService.getJson(`api/watermeters/${meterId}/settings`);

    // Update settings state
    Object.assign(settings, {
      threshold_low: data.threshold_low,
      threshold_high: data.threshold_high,
      threshold_last_low: data.threshold_last_low,
      threshold_last_high: data.threshold_last_high,
      islanding_padding: data.islanding_padding,
      segments: data.segments,
      extended_last_digit: data.extended_last_digit === 1,
      shrink_last_3: data.shrink_last_3 === 1,
      rotated_180: data.rotated_180 === 1,
      max_flow_rate: data.max_flow_rate,
      conf_threshold: data.conf_threshold,
      roi_extractor: data.roi_extractor || 'yolo',
      template_id: data.template_id || null,
      segment_mode: data.segment_mode || 'display',
      use_correctional_alg: data.use_correctional_alg === 1 || data.use_correctional_alg === true
    });

    return data;
  };

  const updateSettings = async (meterId) => {
    const payload = {
      threshold_low: settings.threshold_low,
      threshold_high: settings.threshold_high,
      threshold_last_low: settings.threshold_last_low,
      threshold_last_high: settings.threshold_last_high,
      islanding_padding: settings.islanding_padding,
      rotated_180: settings.rotated_180,
      segments: settings.segments,
      extended_last_digit: settings.extended_last_digit,
      shrink_last_3: settings.shrink_last_3,
      max_flow_rate: settings.max_flow_rate,
      conf_threshold: settings.conf_threshold,
      roi_extractor: settings.roi_extractor,
      template_id: settings.template_id,
      segment_mode: settings.segment_mode,
      use_correctional_alg: settings.use_correctional_alg,
    };

    await apiService.put(`api/watermeters/${meterId}/settings`, payload);
  };

  const fetchSource = async (meterId) => {
    const data = await apiService.getJson('api/sources');
    const meterSource = data.sources.find(s => s.name === meterId);
    source.value = meterSource || null;
    return meterSource;
  };

  const fetchAll = async (meterId) => {
    await Promise.all([
      fetchWatermeter(meterId),
      fetchEvaluations(meterId),
      fetchHistory(meterId),
      fetchSettings(meterId),
      fetchSource(meterId),
    ]);
  };

  return {
    // State
    lastPicture,
    evaluations,
    evaluation,
    history,
    source,
    settings,
    capturing,
    // Actions
    fetchWatermeter,
    fetchEvaluations,
    fetchHistory,
    fetchSettings,
    updateSettings,
    fetchSource,
    fetchAll,
  };
});
