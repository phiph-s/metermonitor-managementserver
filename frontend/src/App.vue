<template>
  <n-space vertical size="large" justify="center">
    <n-layout>
      <n-layout-content content-style="padding: 24px;">
        <div class="app-header">
          <div class="header-bar">
            <div class="header-left">
              <img
                src="@/assets/logo.png"
                alt="Logo"
                class="theme-revert header-logo"
              />
              <div class="nav-divider"></div>
              <router-link to="/" class="nav-item" :class="{ active: isOverviewActive }">
                <n-icon size="15"><HomeOutlined /></n-icon><span v-if="!isMobile">Overview</span>
              </router-link>
              <router-link to="/settings" class="nav-item" :class="{ active: route.path === '/settings' }">
                <n-icon size="15"><SettingsOutlined /></n-icon><span v-if="!isMobile">Settings</span>
              </router-link>
              <router-link v-if="currentMeter" :to="(isSetup ? '/setup/' : '/meter/') + currentMeter" class="nav-item active nav-item--meter">
                <n-icon size="15"><BuildOutlined v-if="isSetup" /><SpeedOutlined v-else /></n-icon><span v-if="!isMobile">{{ currentMeter }}</span>
              </router-link>
            </div>
            <div class="header-right">
            <span class="build-version" :style="{ color: buildVersionColor }">{{ buildTagLabel }}</span>
            <n-popover
              v-if="alertKeys.length > 0"
              trigger="click"
              placement="bottom-end"
              :show-arrow="false"
            >
              <template #trigger>
                <n-button
                  circle
                  quaternary
                  size="small"
                  :style="{ color: alertIconColor }"
                >
                  <template #icon>
                    <n-icon size="18"><WarningAmberOutlined /></n-icon>
                  </template>
                </n-button>
              </template>
              <div class="alert-dropdown">
                <div v-for="key in alertKeys" :key="key" :class="['alert-item', `alert-item--${getAlertType(key, alerts[key])}`]">
                  <span class="alert-key">{{ key }}</span>
                  <span class="alert-msg">{{ alerts[key] }}</span>
                </div>
              </div>
            </n-popover>
            <n-tooltip trigger="hover">
              <template #trigger>
                <n-button
                  circle
                  quaternary
                  size="small"
                  @click="cycleTheme"
                >
                  <template #icon>
                    <n-icon size="18">
                      <LightModeOutlined v-if="!isDark" />
                      <DarkModeOutlined v-else />
                    </n-icon>
                  </template>
                </n-button>
              </template>
              {{ themeTooltip }}
            </n-tooltip>
            <n-button
              v-if="headerState.showRefresh"
              circle
              quaternary
              size="small"
              :loading="headerState.refreshLoading"
              @click="headerState.onRefresh && headerState.onRefresh()"
            >
              <template #icon>
                <n-icon size="18"><RefreshOutlined /></n-icon>
              </template>
            </n-button>
            </div>
          </div>
        </div>
        <router-view></router-view>
        <WhatsNewDialog />
      </n-layout-content>
    </n-layout>
  </n-space>
</template>

<script setup>
import {NLayout, NLayoutContent, NSpace, NButton, NIcon, NTooltip, NPopover, NFlex} from 'naive-ui';
import { LightModeOutlined, DarkModeOutlined, WarningAmberOutlined, RefreshOutlined, HomeOutlined, SettingsOutlined, SpeedOutlined, BuildOutlined } from '@vicons/material';
import {onMounted, onUnmounted, ref, computed, reactive, provide} from "vue";
import WhatsNewDialog from '@/components/WhatsNewDialog.vue';
import { useRoute } from 'vue-router';
import router from "@/router";
import { useThemeStore } from '@/stores/themeStore';
import { storeToRefs } from 'pinia';
import { headerControlsKey } from '@/composables/headerControls';

const themeStore = useThemeStore();
const { isDark, themeMode, isHomeAssistant } = storeToRefs(themeStore);
const route = useRoute();

const headerState = reactive({
  showRefresh: false,
  refreshLoading: false,
  onRefresh: null
});

const setHeader = (next) => {
  Object.assign(headerState, next);
};

const resetHeader = () => {
  headerState.showRefresh = false;
  headerState.refreshLoading = false;
  headerState.onRefresh = null;
};

provide(headerControlsKey, { headerState, setHeader, resetHeader });

const themeTooltip = computed(() => {
  if (themeMode.value === 'auto') {
    return isHomeAssistant.value
      ? 'Auto (synced with Home Assistant)'
      : 'Auto (follows system)';
  }
  return themeMode.value === 'dark' ? 'Dark mode' : 'Light mode';
});

const cycleTheme = () => {
  // Cycle: auto -> light -> dark -> auto
  const modes = ['auto', 'light', 'dark'];
  const currentIndex = modes.indexOf(themeMode.value);
  const nextIndex = (currentIndex + 1) % modes.length;
  themeStore.setThemeMode(modes[nextIndex]);
};

const isOverviewActive = computed(() => route.path === '/' || route.path === '/list');
const currentMeter = computed(() => (route.name === 'Meter' || route.name === 'Setup') ? route.params.id : null);
const isSetup = computed(() => route.name === 'Setup');
const isMobile = ref(window.innerWidth < 600);
const updateMobile = () => { isMobile.value = window.innerWidth < 600; };
onMounted(() => window.addEventListener('resize', updateMobile));
onUnmounted(() => window.removeEventListener('resize', updateMobile));

const alerts = ref({});
const alertKeys = computed(() => Object.keys(alerts.value));

const getAlertType = (key, message) => {
  if (key === 'authentication') return 'warning';
  if (key === 'mqtt' && message === 'Connecting to MQTT broker') return 'info';
  return 'error';
};

const pillType = computed(() => {
  for (const key of alertKeys.value) {
    if (getAlertType(key, alerts.value[key]) === 'error') return 'error';
  }
  for (const key of alertKeys.value) {
    if (getAlertType(key, alerts.value[key]) === 'warning') return 'warning';
  }
  return 'info';
});

const alertIconColor = computed(() => {
  const t = pillType.value;
  if (t === 'error') return '#d03050';
  if (t === 'warning') return '#f0a020';
  return '#2080f0';
});

const host = import.meta.env.VITE_HOST;
const websocket = ref(null);
const reconnectTimer = ref(null);
const websocketStopped = ref(false);

const buildTagLabel = computed(() => {
  const devBranch = (__GIT_BRANCH__ || '').toLowerCase().includes('dev');
  if (import.meta.env.DEV || devBranch) {
    return (__GIT_COMMIT__ || 'dev').slice(0, 8);
  }
  return `v${__APP_VERSION__ || '0.0.0'}`;
});

const buildVersionColor = computed(() => {
  const devBranch = (__GIT_BRANCH__ || '').toLowerCase().includes('dev');
  return (import.meta.env.DEV || devBranch) ? '#f0a020' : '#18a058';
});

const toWebsocketUrl = (baseHost) => {
  const apiProbe = new URL(`${baseHost}api/alerts`, window.location.href);
  const wsTarget = new URL(apiProbe.toString());
  wsTarget.pathname = wsTarget.pathname.replace(/api\/alerts$/, 'api/ws/evaluations');
  wsTarget.protocol = wsTarget.protocol === 'https:' ? 'wss:' : 'ws:';
  const secret = encodeURIComponent(localStorage.getItem('secret') || '');
  wsTarget.search = `secret=${secret}`;
  return wsTarget.toString();
};

const connectWebsocket = () => {
  if (reconnectTimer.value) {
    clearTimeout(reconnectTimer.value);
    reconnectTimer.value = null;
  }
  if (websocket.value) {
    websocket.value.close();
  }
  const wsUrl = toWebsocketUrl(host);
  const ws = new WebSocket(wsUrl);
  websocket.value = ws;

  ws.onmessage = (_event) => {
    try {
      const payload = JSON.parse(_event.data);
      if (payload?.type === 'evaluation_created') {
        window.dispatchEvent(new CustomEvent('meter-evaluation-updated', { detail: payload }));
      } else if (payload?.type === 'alerts_updated') {
        alerts.value = payload.alerts ?? {};
      }
    } catch (_err) {
      // Ignore malformed websocket payloads.
    }
  };

  ws.onclose = () => {
    if (websocketStopped.value) return;
    reconnectTimer.value = setTimeout(connectWebsocket, 3000);
  };

  ws.onerror = () => {
    ws.close();
  };
};

const fetchInitialAlerts = async () => {
  try {
    const r = await fetch(host + 'api/alerts', {
      headers: {secret: localStorage.getItem('secret')}
    });
    if (r.status === 401) {
      await router.push({path: '/unlock'});
      return;
    }
    alerts.value = await r.json();
  } catch (_err) {
    // Will be updated via WebSocket once connected
  }
};

onMounted(() => {
  websocketStopped.value = false;
  fetchInitialAlerts();
  connectWebsocket();
});

onUnmounted(() => {
  websocketStopped.value = true;
  if (reconnectTimer.value) clearTimeout(reconnectTimer.value);
  if (websocket.value) websocket.value.close();
});

</script>
<style>

.apexcharts-tooltip {
  background: #f3f3f3;
  color: #292929;
}

.google-sans-code {
  font-family: "Google Sans Code", monospace;
  font-optical-sizing: auto;
  font-weight: 600;
  font-style: normal;
}

.light-mode .theme-revert {
  mix-blend-mode: multiply;
  filter: invert(1);
}

.app-header {
  margin-bottom: 16px;
}

.header-bar {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
}

.light-mode .header-bar {
  background: rgba(0, 0, 0, 0.08);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12), 0 1px 4px rgba(0, 0, 0, 0.08);
}

.header-left {
  display: flex;
  align-items: stretch;
  gap: 2px;
}

.nav-divider {
  width: 1px;
  background: rgba(255, 255, 255, 0.1);
  margin: 6px 10px;
  flex-shrink: 0;
}

.light-mode .nav-divider {
  background: rgba(0, 0, 0, 0.12);
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 0 12px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  color: inherit;
  opacity: 0.7;
  transition: background 0.15s, opacity 0.15s;
  white-space: nowrap;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.07);
  opacity: 1;
}

.light-mode .nav-item:hover {
  background: rgba(0, 0, 0, 0.06);
}

.nav-item.active {
  background: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
  opacity: 1;
}

.light-mode .nav-item.active {
  background: rgba(59, 130, 246, 0.1);
}

.nav-item--meter {
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 600px) {
  .nav-item {
    padding: 0 8px;
  }
  .header-logo {
    max-width: 70px;
  }
}

.header-right {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.header-logo {
  max-width: 100px;
  align-self: center;
}

.n-dialog{
  border-radius: 12px;
}

.build-version {
  font-size: 11px;
  font-weight: 600;
  font-family: monospace;
  opacity: 0.85;
}

.alert-dropdown {
  min-width: 220px;
  max-width: 340px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.alert-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px 8px;
  border-radius: 6px;
  border-left: 3px solid transparent;
}

.alert-item--error {
  border-left-color: #d03050;
  background: rgba(208, 48, 80, 0.08);
}

.alert-item--warning {
  border-left-color: #f0a020;
  background: rgba(240, 160, 32, 0.08);
}

.alert-item--info {
  border-left-color: #2080f0;
  background: rgba(32, 128, 240, 0.08);
}

.alert-key {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  opacity: 0.6;
  letter-spacing: 0.05em;
}

.alert-msg {
  font-size: 13px;
  line-height: 1.4;
}

</style>
