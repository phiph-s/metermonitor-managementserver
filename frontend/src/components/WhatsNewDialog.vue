<template>
  <n-modal
    v-model:show="show"
    :mask-closable="false"
    style="max-width: 580px; width: 92vw;"
    :bordered="false"
    transform-origin="center"
  >
    <n-card :bordered="false" class="whats-new-card">
      <div class="wn-header">
        <n-tag type="success" size="small" round class="wn-version-tag">v5.0</n-tag>
        <h2 class="wn-title">What's new in MeterMonitor</h2>
        <p class="wn-subtitle">Here's a summary of what changed in this major release.</p>
      </div>

      <div class="wn-features">
        <div class="wn-feature" v-for="f in features" :key="f.title">
          <div class="wn-icon">
            <n-icon size="20" :component="f.icon" />
          </div>
          <div class="wn-text">
            <div class="wn-feature-title">{{ f.title }}</div>
            <div class="wn-feature-desc">{{ f.desc }}</div>
          </div>
        </div>
      </div>

      <div class="wn-footer">
        <n-button type="primary" round size="large" @click="dismiss">
          Got it
        </n-button>
      </div>
    </n-card>
  </n-modal>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { NModal, NCard, NButton, NTag, NIcon } from 'naive-ui';
import {
  SettingsOutlined,
  ShowChartOutlined,
  WifiTetheringOutlined,
  SpeedOutlined,
  CalendarTodayOutlined,
  TuneOutlined,
} from '@vicons/material';

const VERSION_KEY = 'whats_new_seen';
const CURRENT_VERSION = '5.0';

const show = ref(false);

const features = [
  {
    icon: SpeedOutlined,
    title: 'Meter type & unit support',
    desc: 'Meters can be configured as Water, Gas, Electricity or custom types. Each type has a default unit and color — shown across cards, charts and evaluation details.',
  },
  {
    icon: ShowChartOutlined,
    title: 'Daily history & consumption charts',
    desc: 'A new daily history table records one snapshot per day and is never pruned. The meter view now shows both a detailed and a daily chart with average consumption.',
  },
  {
    icon: CalendarTodayOutlined,
    title: 'Consumption stats on meter cards',
    desc: 'Each meter card now shows today\'s consumption, the daily average, and an extrapolated yearly figure — replacing the previous sparkline.',
  },
  {
    icon: TuneOutlined,
    title: 'Redesigned navigation',
    desc: 'The topbar now shows contextual navigation items — Overview, Settings, and the current meter name — with smooth transitions and active-state highlighting.',
  },
  {
    icon: SettingsOutlined,
    title: 'In-app settings',
    desc: 'MQTT credentials, broker configuration, and history retention limits can now be changed directly in the UI — no more editing options.json in Home Assistant.',
  },
  {
    icon: WifiTetheringOutlined,
    title: 'Improved MQTT reliability',
    desc: 'Wrong credentials now surface as a clear error alert. Reconnect attempts use exponential backoff and only clear the alert once the broker actually accepts the connection.',
  }
];

onMounted(() => {
  if (localStorage.getItem(VERSION_KEY) !== CURRENT_VERSION) {
    show.value = true;
  }
});

const dismiss = () => {
  localStorage.setItem(VERSION_KEY, CURRENT_VERSION);
  show.value = false;
};
</script>

<style scoped>
.whats-new-card {
  border-radius: 16px !important;
  overflow: hidden;
}

.wn-header {
  text-align: center;
  padding: 8px 0 20px;
}

.wn-version-tag {
  margin-bottom: 10px;
}

.wn-title {
  margin: 0 0 6px;
  font-size: 1.35em;
  font-weight: 700;
  line-height: 1.2;
}

.wn-subtitle {
  margin: 0;
  opacity: 0.55;
  font-size: 0.9em;
}

.wn-features {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 24px;
}

.wn-feature {
  display: flex;
  align-items: flex-start;
  gap: 14px;
}

.wn-icon {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  background: rgba(8, 145, 178, 0.12);
  color: #0891b2;
  display: flex;
  align-items: center;
  justify-content: center;
}

.wn-text {
  flex: 1;
  min-width: 0;
}

.wn-feature-title {
  font-weight: 600;
  font-size: 0.95em;
  margin-bottom: 2px;
}

.wn-feature-desc {
  font-size: 0.85em;
  opacity: 0.65;
  line-height: 1.45;
}

.wn-footer {
  display: flex;
  justify-content: flex-end;
  padding-top: 4px;
}
</style>
