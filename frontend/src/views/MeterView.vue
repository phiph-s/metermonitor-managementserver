<template>
  <template v-if="isMobile">
    <n-tabs type="line" animated>
      <n-tab-pane name="details" tab="Details">
        <div v-if="loading" class="meter-skeleton">
          <n-skeleton height="32px" width="55%" />
          <n-skeleton class="skeleton-image" />
          <n-skeleton height="16px" width="80%" />
          <n-skeleton height="16px" width="70%" />
          <n-skeleton height="16px" width="60%" />
        </div>
        <MeterDetails
          v-else
          :data="data"
          :settings="settings"
          :id="id"
          :history="history"
          :daily-history="dailyHistory"
          :downloadingDataset="downloadingDataset"
          :capabilities="data?.capabilities ?? []"
          @resetToSetup="resetToSetup"
          @triggerCapture="triggerCapture"
          @deleteMeter="deleteMeter"
          @clearEvaluations="clearEvaluations"
          @downloadDataset="downloadDataset"
          @deleteDataset="deleteDataset"
        />
      </n-tab-pane>
      <n-tab-pane name="evaluations" tab="Evaluations">
        <div v-if="loading" class="evaluations-skeleton">
          <n-skeleton height="20px" width="40%" />
          <div v-for="i in 4" :key="i" class="evaluations-skeleton-card">
            <n-skeleton height="16px" width="30%" />
            <n-skeleton height="12px" width="90%" />
            <n-skeleton height="12px" width="75%" />
          </div>
        </div>
        <div v-else-if="evaluations !== null" style="padding-left: 10px; padding-right: 10px;">
          <EvaluationResultList :evaluations="evaluations" :name="id" :decimals="settings.decimals" :meter-type="settings.meter_type" :unit="settings.unit" @load-more="loadMoreEvaluations" @dataset-uploaded="loadMeter" :has-more="hasMoreEvaluations"/>
        </div>
      </n-tab-pane>
      <n-tab-pane name="evaluations" tab="Statistics">

      </n-tab-pane>
    </n-tabs>
  </template>

  <template v-else>
    <div class="meter-layout">
      <aside class="meter-sidebar">
        <div class="sidebar-content">
          <div v-if="loading" class="meter-skeleton">
            <n-skeleton height="32px" width="55%" />
            <n-skeleton class="skeleton-image" />
            <n-skeleton height="16px" width="80%" />
            <n-skeleton height="16px" width="70%" />
            <n-skeleton height="16px" width="60%" />
          </div>
          <MeterDetails
            v-else
            :data="data"
            :settings="settings"
            :id="id"
            :downloadingDataset="downloadingDataset"
            :history="history"
            :daily-history="dailyHistory"
            :capabilities="data?.capabilities ?? []"
            @resetToSetup="resetToSetup"
            @triggerCapture="triggerCapture"
            @deleteMeter="deleteMeter"
            @clearEvaluations="clearEvaluations"
            @downloadDataset="downloadDataset"
            @deleteDataset="deleteDataset"
          />
        </div>
      </aside>
      <n-tabs type="line" animated>
        <n-tab-pane name="details" tab="Evaluations">
          <main class="meter-content" v-if="loading">
            <div class="evaluations-skeleton">
              <n-skeleton height="20px" width="40%" />
              <div v-for="i in 6" :key="i" class="evaluations-skeleton-card">
                <n-skeleton height="16px" width="30%" />
                <n-skeleton height="12px" width="90%" />
                <n-skeleton height="12px" width="75%" />
              </div>
            </div>
          </main>
          <main class="meter-content" v-else-if="evaluations !== null">
            <EvaluationResultList :evaluations="evaluations" :name="id" :decimals="settings.decimals" :meter-type="settings.meter_type" :unit="settings.unit" @load-more="loadMoreEvaluations" @dataset-uploaded="loadMeter" :has-more="hasMoreEvaluations"/>
          </main>
        </n-tab-pane>
        <n-tab-pane name="stats" tab="Statistics">
        </n-tab-pane>
      </n-tabs>
    </div>
  </template>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import router from '@/router';
import EvaluationResultList from "@/components/EvaluationResultList.vue";
import MeterDetails from "@/components/MeterDetails.vue";
import { NSkeleton, NTabs, NTabPane, useMessage } from "naive-ui";
import { useWatermeterStore } from '@/stores/watermeterStore';
import { storeToRefs } from 'pinia';
import { useHeaderControls } from '@/composables/headerControls';

const route = useRoute();
const id = computed(() => route.params.id);
const store = useWatermeterStore();
const { lastPicture: data, evaluations, history, dailyHistory, settings } = storeToRefs(store);

const loading = ref(false);
const refreshing = ref(false);
const downloadingDataset = ref(false);
const hasMoreEvaluations = ref(true);
const isMobile = ref(window.innerWidth < 1000);
const headerControls = useHeaderControls();
let evaluationEventHandler = null;

const updateWidth = () => {
  isMobile.value = window.innerWidth < 800;
};

onMounted(() => {
  window.addEventListener('resize', updateWidth);
  evaluationEventHandler = (event) => {
    const meterName = event?.detail?.name;
    if (meterName && meterName === id.value) {
      refreshMeter();
    }
  };
  window.addEventListener('meter-evaluation-updated', evaluationEventHandler);
  if (headerControls) {
    headerControls.setHeader({
      showRefresh: true,
      onRefresh: refreshMeter,
      refreshLoading: refreshing.value
    });
  }
});

onUnmounted(() => {
  window.removeEventListener('resize', updateWidth);
  if (evaluationEventHandler) {
    window.removeEventListener('meter-evaluation-updated', evaluationEventHandler);
  }
  if (headerControls) {
    headerControls.resetHeader();
  }
});

watch(refreshing, (next) => {
  if (!headerControls) return;
  headerControls.setHeader({ refreshLoading: next });
});

const host = import.meta.env.VITE_HOST;

// Initial load: resets data and shows skeletons (used on mount / meter change)
const loadMeter = async () => {
  loading.value = true;
  hasMoreEvaluations.value = true;
  store.resetMeterData();
  try {
    await store.fetchAll(id.value);
  } catch (e) {
    if (e.response && e.response.status === 401) {
      router.push({ path: '/unlock' });
    }
  }
  loading.value = false;
};

// Background refresh: keeps components mounted, updates data silently
const refreshMeter = async () => {
  refreshing.value = true;
  try {
    await store.fetchAll(id.value);
  } catch (e) {
    if (e.response && e.response.status === 401) {
      router.push({ path: '/unlock' });
    }
  }
  refreshing.value = false;
};

watch(
  () => route.params.id,
  () => {
    loadMeter();
  },
  { immediate: true }
);

const loadMoreEvaluations = async () => {
  if (!evaluations.value || evaluations.value.length === 0) return;
  const lastId = evaluations.value[evaluations.value.length - 1].id;
  const data = await store.fetchEvaluations(id.value, 10, lastId);
  if (!data?.evals?.length || data.evals.length < 10) {
    hasMoreEvaluations.value = false;
  }
};

const deleteMeter = async () => {
  let response = await fetch(host + 'api/watermeters/' + id.value, {
    method: 'DELETE',
    headers: { secret: localStorage.getItem('secret') }
  });
  if (response.status === 200) {
    router.replace({ path: '/' });
  } else {
    console.log('Error deleting meter');
  }
};

const resetToSetup = async () => {
  let response = await fetch(host + 'api/setup/' + id.value + '/enable', {
    method: 'POST',
    headers: { secret: localStorage.getItem('secret') }
  });
  if (response.status === 200) {
    router.replace({ path: '/setup/' + id.value });
  } else {
    console.log('Error resetting meter');
  }
};

const message = useMessage();

const triggerCapture = async () => {
  try {
    store.capturing = true;
    const response = await fetch(host + 'api/sources/' + store.source.id + '/capture', {
      method: 'POST',
      headers: { secret: localStorage.getItem('secret') }
    });

    if (!response.ok) {
      const detail = (await response.json()).detail;
      const isTimeout = response.status === 408;
      message.error(isTimeout ? 'Capture timed out — no image received within 5 seconds' : 'Error triggering capture: ' + detail, {
        closable: true,
        duration: isTimeout ? 8000 : 60000
      });
      return;
    }

    await refreshMeter();
  } catch (err) {
    message.error('Error triggering capture: ' + err.message);
  } finally {
    store.capturing = false;
  }
};

const downloadDataset = async () => {
  downloadingDataset.value = true;
  try {
    const response = await fetch(host + 'api/dataset/' + id.value + '/download', {
      headers: { secret: localStorage.getItem('secret') }
    });

    if (response.status === 200) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${id.value}_dataset.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } else {
      console.log('Error downloading dataset');
    }
  } catch (err) {
    console.log('Error downloading dataset:', err);
  } finally {
    downloadingDataset.value = false;
  }
};

const deleteDataset = async () => {
  try {
    const response = await fetch(host + 'api/dataset/' + id.value, {
      method: 'DELETE',
      headers: { secret: localStorage.getItem('secret') }
    });

    if (response.status === 200) {
      // Reload meter data to update dataset_present status
      await refreshMeter();
    } else {
      console.log('Error deleting dataset');
    }
  } catch (err) {
    console.log('Error deleting dataset:', err);
  }
};

const clearEvaluations = async () => {
  try {
    const response = await fetch(host + 'api/watermeters/' + id.value + '/evals', {
      method: 'DELETE',
      headers: { secret: localStorage.getItem('secret') }
    });

    if (response.status === 200) {
      const result = await response.json();
      console.log(`Cleared ${result.count} evaluations`);

      // Re-evaluate latest picture to restore state
      await fetch(host + 'api/watermeters/' + id.value + '/evaluations/reevaluate', {
        method: 'POST',
        headers: { secret: localStorage.getItem('secret') }
      });

      // Reload meter data to update evaluations
      await refreshMeter();
    } else {
      console.log('Error clearing evaluations');
    }
  } catch (err) {
    console.log('Error clearing evaluations:', err);
  }
};
</script>

<style scoped>
.meter-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
  min-height: calc(100vh - 140px);
}

.meter-sidebar {
  width: 380px;
  flex: 0 0 380px;
  min-width: 380px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}

.sidebar-content {
  width: 100%;
}

.meter-content {
  flex: 1;
  min-width: 0;
  height: calc(100vh - 200px);
}

.meter-skeleton {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 12px;
}

.skeleton-image {
  height: 220px;
  width: 100%;
  border-radius: 12px;
}

.evaluations-skeleton {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 12px;
}

.evaluations-skeleton-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.04);
}

.light-mode .meter-sidebar {
  background: rgba(0, 0, 0, 0.08);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12), 0 1px 4px rgba(0, 0, 0, 0.08);
}

.light-mode .evaluations-skeleton-card {
  background: rgba(0, 0, 0, 0.04);
}
</style>
