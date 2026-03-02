import { defineStore } from 'pinia';
import { ref } from 'vue';
import { apiService } from '@/services/api';
import { useWatermeterStore } from './watermeterStore';

export const useSetupStore = defineStore('setup', () => {
  // State
  const currentStep = ref(1);
  const randomExamples = ref([]);
  const noBoundingBox = ref(false);
  const loading = ref(false);
  const loadingCancelled = ref(false);
  const runningBenchmark = ref(false);
  const capturing = ref(false);
  const searchingThresholds = ref(false);
  const thresholdSearchResult = ref(null);
  const tooFewEvaluations = ref(false);
  const templateSaving = ref(false);
  const templateData = ref(null);
  const reevaluateError = ref(null);

  // Actions
  const searchThresholds = async (meterId, steps = 10) => {
    const watermeterStore = useWatermeterStore();

    searchingThresholds.value = true;
    thresholdSearchResult.value = null;

    try {
      const response = await apiService.post(`api/watermeters/${meterId}/search_thresholds`, { steps });

      if (response.ok) {
        const result = await response.json();

        if (result.error) {
          console.error('Threshold search error:', result.error);
          thresholdSearchResult.value = { error: result.error };
          return null;
        }

        thresholdSearchResult.value = result;

        // Apply the found thresholds to settings
        watermeterStore.settings.threshold_low = result.threshold[0];
        watermeterStore.settings.threshold_high = result.threshold[1];
        watermeterStore.settings.threshold_last_low = result.threshold_last[0];
        watermeterStore.settings.threshold_last_high = result.threshold_last[1];

        // Save settings to backend
        await watermeterStore.updateSettings(meterId);

        console.log('Threshold search completed:', result);
        return result;
      } else {
        console.error('Threshold search failed:', response.status);
        thresholdSearchResult.value = { error: `Request failed: ${response.status}` };
        return null;
      }
    } catch (e) {
      console.error('Threshold search exception:', e);
      thresholdSearchResult.value = { error: e.message };
      return null;
    } finally {
      searchingThresholds.value = false;
    }
  };

  const nextStep = (step) => {
    if (step === 1) {
      currentStep.value = 2;
    } else if (step === 2) {
      currentStep.value = 3;
    }
  };

  const setLoading = (value) => {
    loading.value = value;
  };

  const updateThresholds = async (data, meterId) => {
    const watermeterStore = useWatermeterStore();
    
    watermeterStore.settings.threshold_low = data.threshold[0];
    watermeterStore.settings.threshold_high = data.threshold[1];
    watermeterStore.settings.threshold_last_low = data.threshold_last[0];
    watermeterStore.settings.threshold_last_high = data.threshold_last[1];
    watermeterStore.settings.islanding_padding = data.islanding_padding;

    // Cancel any ongoing loading and clear random examples
    loadingCancelled.value = true;
    randomExamples.value = [];

    await watermeterStore.updateSettings(meterId);
  };

  const updateMaxFlow = async (value, meterId) => {
    const watermeterStore = useWatermeterStore();
    
    watermeterStore.settings.max_flow_rate = value;
    await watermeterStore.updateSettings(meterId);
  };

  const updateConfThreshold = async (value, meterId) => {
    const watermeterStore = useWatermeterStore();

    watermeterStore.settings.conf_threshold = value;
    await watermeterStore.updateSettings(meterId);
  }

  const updateUseCorrection = async (value, meterId) => {
    const watermeterStore = useWatermeterStore();

    watermeterStore.settings.use_correctional_alg = value;
    await watermeterStore.updateSettings(meterId);
  }

  const updateDigitModels = async (models, meterId) => {
    const watermeterStore = useWatermeterStore();

    watermeterStore.settings.digit_models = Array.isArray(models) ? models : null;
    await watermeterStore.updateSettings(meterId);
  };

  const updateSegmentationSettings = async (data, meterId) => {
    const watermeterStore = useWatermeterStore();

    // Cancel any ongoing loading
    loadingCancelled.value = true;

    const previousExtractor = watermeterStore.settings.roi_extractor || 'yolo';
    const nextExtractor = data.roiExtractor || previousExtractor;
    const isTemplateExtractor = (value) => ['orb', 'static_rect'].includes(value);
    const previousSegmentMode = watermeterStore.settings.segment_mode || 'display';
    const nextSegmentMode = data.segmentMode || previousSegmentMode;

    watermeterStore.settings.segments = data.segments;
    watermeterStore.settings.extended_last_digit = data.extendedLastDigit;
    watermeterStore.settings.shrink_last_3 = data.last3DigitsNarrow;
    watermeterStore.settings.rotated_180 = data.rotated180;
    watermeterStore.settings.roi_extractor = nextExtractor;
    watermeterStore.settings.segment_mode = nextSegmentMode;

    if (nextExtractor !== previousExtractor || nextSegmentMode !== previousSegmentMode) {
      if (!isTemplateExtractor(nextExtractor) || previousExtractor !== nextExtractor || nextSegmentMode !== previousSegmentMode) {
        watermeterStore.settings.template_id = null;
        templateData.value = null;
      }
    }

    await watermeterStore.updateSettings(meterId);
    if (!isTemplateExtractor(nextExtractor) || watermeterStore.settings.template_id) {
      await reevaluate(meterId);
    }
  };

  const clearEvaluationExamples = (meterId=null) => {
    randomExamples.value = [];
    if (meterId) requestReevaluatedDigits(meterId)
  }

  const reevaluate = async (meterId) => {
    // Cancel any ongoing loading
    loadingCancelled.value = true;

    loading.value = true;
    reevaluateError.value = null;
    try {
      const response = await apiService.post(`api/watermeters/${meterId}/evaluations/reevaluate`);

      if (response.ok) {
        const result = await response.json();

        if (result.error) {
          console.error('reevaluate error', result.error);
          reevaluateError.value = result.error;
          return;
        }

        noBoundingBox.value = !result["result"]
        if (!result["result"] && result.error) {
          reevaluateError.value = result.error;
        }
      }
    } catch (e) {
      console.error('reevaluate failed', e);
      reevaluateError.value = e.message || 'Re-evaluation failed';
    } finally {
      // Refresh the data
      const watermeterStore = useWatermeterStore();
      await watermeterStore.fetchAll(meterId);
      loading.value = false;
      clearEvaluationExamples();
    }
  };

  const requestReevaluatedDigits = async (meterId, maxAmount = 10) => {
    // Clear existing examples and reset cancellation flag
    randomExamples.value = [];
    loadingCancelled.value = false;
    runningBenchmark.value = true;
    tooFewEvaluations.value = false;

    try {
      // First, get the count of available evaluations
      const countResponse = await apiService.get(`api/watermeters/${meterId}/evals/count`);
      if (!countResponse.ok) {
        console.error('Failed to get evaluation count');
        runningBenchmark.value = false;
        return;
      }

      const countResult = await countResponse.json();
      const availableCount = countResult.count;

      // Need at least 2 evaluations to run benchmark (1 is the current one)
      if (availableCount < 2) {
        console.log('Too few evaluations for benchmark');
        tooFewEvaluations.value = true;
        runningBenchmark.value = false;
        return;
      }

      // Limit amount to available evaluations (minus 1 for current)
      const amount = Math.min(maxAmount, availableCount - 1);
      const usedOffsets = new Set();

      for (let i = 0; i < amount; i++) {
        // Check if loading was cancelled
        if (loadingCancelled.value) {
          console.log('Sample loading cancelled');
          break;
        }

        const url = `api/watermeters/${meterId}/evaluations/sample/-1`;
        const response = await apiService.post(url);

        if (response.ok) {
          const result = await response.json();

          if (result.error) {
            console.error('get_reevaluated_digits error', result.error);
            break;
          }

          // Only add if not cancelled
          if (!loadingCancelled.value) {
            randomExamples.value.push(result);
          }
        } else {
          console.error('Failed to fetch sample');
          break;
        }
      }
    } catch (e) {
      console.error('get_reevaluated_digits failed', e);
    } finally {
      runningBenchmark.value = false;
    }
  };

  const redoDigitEval = async (meterId) => {
    // Clear existing examples and reset cancellation flag
    loading.value = true;

    const url = `api/watermeters/${meterId}/evaluations/sample`;
    const response = await apiService.post(url);

    if (response.ok) {
      const result = await response.json();

      if (result.error) {
        console.error('redoDigitEval error', result.error);
        loading.value = false;
        return;
      }

      const watermeterStore = useWatermeterStore();
      watermeterStore.evaluation.th_digits = result.processed_images;
      watermeterStore.evaluation.predictions = result.predictions;

      loading.value = false;

    } else {
      console.error('Failed to redo digit evaluation');
      loading.value = false;
    }
  };

  const getData = async (meterId) => {
    loading.value = true;
    const watermeterStore = useWatermeterStore();
    await watermeterStore.fetchAll(meterId);
    loading.value = false;
  };

  const fetchTemplate = async (templateId) => {
    if (!templateId) {
      templateData.value = null;
      return null;
    }
    try {
      templateData.value = await apiService.getJson(`api/templates/${templateId}`);
      return templateData.value;
    } catch (e) {
      console.error('Failed to load template', e);
      templateData.value = null;
      return null;
    }
  };

  const saveTemplate = async (meterId, points, digitQuads) => {
    if (templateSaving.value) return;
    templateSaving.value = true;
    try {
      const watermeterStore = useWatermeterStore();
      if (!['orb', 'static_rect'].includes(watermeterStore.settings.roi_extractor)) {
        console.error('Current extractor does not require a template');
        return;
      }
      const picture = watermeterStore.lastPicture?.picture;
      if (!picture?.data) {
        console.error('No reference image available');
        return;
      }
      const segmentMode = watermeterStore.settings.segment_mode || 'display';
      let normalizedPoints = Array.isArray(points) && points.length === 4 ? points : null;
      if (!normalizedPoints && templateData.value?.config?.display_corners?.length === 4) {
        const corners = templateData.value.config.display_corners;
        const imageWidth = templateData.value.image_width || 1;
        const imageHeight = templateData.value.image_height || 1;
        normalizedPoints = corners.map((point) => ({
          x: point[0] / imageWidth,
          y: point[1] / imageHeight
        }));
      }
      if (!normalizedPoints && segmentMode === 'display') {
        normalizedPoints = [
          { x: 0.2, y: 0.2 },
          { x: 0.8, y: 0.2 },
          { x: 0.8, y: 0.8 },
          { x: 0.2, y: 0.8 }
        ];
      }

      let normalizedDigitQuads = null;
      if (segmentMode === 'each_digit') {
        if (Array.isArray(digitQuads) && digitQuads.length > 0) {
          normalizedDigitQuads = digitQuads;
        } else if (templateData.value?.config?.digit_corners?.length) {
          const corners = templateData.value.config.digit_corners;
          const imageWidth = templateData.value.image_width || 1;
          const imageHeight = templateData.value.image_height || 1;
          normalizedDigitQuads = corners.map((quad) => quad.map((point) => ({
            x: point[0] / imageWidth,
            y: point[1] / imageHeight
          })));
        }
        if (!normalizedDigitQuads) {
          console.error('No digit quads available for each_digit mode');
          return;
        }
        if (normalizedDigitQuads.length !== watermeterStore.settings.segments) {
          console.error('Digit quad count does not match segments');
          return;
        }
      }
      const payload = {
        name: meterId,
        extractor: watermeterStore.settings.roi_extractor,
        reference_image_base64: picture.data,
        image_width: picture.width,
        image_height: picture.height,
        segment_mode: segmentMode,
      };
      if (normalizedPoints) {
        payload.display_corners = normalizedPoints.map((point) => [point.x, point.y]);
      } else {
        payload.display_corners = [];
      }
      if (normalizedDigitQuads) {
        payload.digit_corners = normalizedDigitQuads.map((quad) =>
          quad.map((point) => [point.x ?? point[0], point.y ?? point[1]])
        );
      }
      const result = await apiService.postJson('api/templates', payload);
      if (!result?.id) {
        console.error('Template creation failed');
        return;
      }
      watermeterStore.settings.template_id = result.id;
      await watermeterStore.updateSettings(meterId);
      await fetchTemplate(result.id);
      await reevaluate(meterId);
    } catch (e) {
      console.error('Failed to save template', e);
    } finally {
      templateSaving.value = false;
    }
  };

  const triggerCapture = async (meterId) => {
    if (capturing.value) return;
    capturing.value = true;
    try {
      const watermeterStore = useWatermeterStore();
      if (!watermeterStore.source || !watermeterStore.source.id) {
        await watermeterStore.fetchSource(meterId);
      }
      const source = watermeterStore.source;
      if (!source || !source.id) {
        console.error('No source available for capture');
        return;
      }
      const response = await apiService.post(`api/sources/${source.id}/capture`);
      if (!response.ok) {
        console.error('Capture failed');
        return;
      }
      await watermeterStore.fetchAll(meterId);
    } catch (e) {
      console.error('Capture failed', e);
    } finally {
      capturing.value = false;
    }
  };

  const reset = () => {
    console.log('Resetting setup store state');
    currentStep.value = 1;
    randomExamples.value = [];
    noBoundingBox.value = false;
    loading.value = false;
    loadingCancelled.value = false;
    searchingThresholds.value = false;
    thresholdSearchResult.value = null;
    tooFewEvaluations.value = false;
    templateSaving.value = false;
    templateData.value = null;
    reevaluateError.value = null;
  }

  return {
    // State
    currentStep,
    randomExamples,
    noBoundingBox,
    loading,
    capturing,
    runningBenchmark,
    searchingThresholds,
    thresholdSearchResult,
    tooFewEvaluations,
    templateSaving,
    templateData,
    reevaluateError,
    // Actions
    reset,
    redoDigitEval,
    nextStep,
    setLoading,
    searchThresholds,
    updateThresholds,
    updateMaxFlow,
    updateConfThreshold,
    updateUseCorrection,
    updateDigitModels,
    updateSegmentationSettings,
    clearEvaluationExamples,
    reevaluate,
    requestReevaluatedDigits,
    getData,
    triggerCapture,
    fetchTemplate,
    saveTemplate,
  };
});
