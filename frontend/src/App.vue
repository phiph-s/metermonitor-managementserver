<template>

  <n-space vertical size="large">
    <n-layout>
      <n-layout-content content-style="padding: 24px;">
        <div class="app-header">
          <div class="header-bar">
            <div class="header-left">
              <transition name="back-slide" mode="out-in">
                <router-link v-if="showBack" to="/" key="back" class="back-btn-wrapper">
                  <n-button quaternary round size="large" style="padding: 0; font-size: 16px;">
                    ← Back
                  </n-button>
                </router-link>
              </transition>
              <img
                src="@/assets/logo.png"
                alt="Logo"
                class="theme-revert header-logo"
              />
            </div>
            <div class="header-right">
            <n-tag size="small" :type="buildTagType" class="build-tag">
              {{ buildTagLabel }}
            </n-tag>
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
import {NLayout, NLayoutContent, NSpace, NButton, NIcon, NTooltip, NTag, NPopover} from 'naive-ui';
import { LightModeOutlined, DarkModeOutlined, WarningAmberOutlined, RefreshOutlined } from '@vicons/material';
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

const showBack = computed(() => route.path !== '/');

const alerts = ref({});
const alertKeys = computed(() => Object.keys(alerts.value));

const getAlertType = (key, message) => {
  if (key === 'authentication') return 'warning';
  if (key === 'mqtt' && message?.toLowerCase().includes('connecting')) return 'info';
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

const buildTagType = computed(() => {
  const devBranch = (__GIT_BRANCH__ || '').toLowerCase().includes('dev');
  return (import.meta.env.DEV || devBranch) ? 'warning' : 'success';
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

.back-btn-wrapper {
  display: inline-flex;
  text-decoration: none;
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
  align-items: center;
  gap: 12px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 4px;
}

.header-logo {
  max-width: 100px;
}

.n-dialog{
  border-radius: 12px;
}

.build-tag {
  margin-right: 4px;
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

.back-slide-enter-active,
.back-slide-leave-active {
  transition: all 0.25s ease;
  overflow: hidden;
  max-width: 120px;
}

.back-slide-enter-from,
.back-slide-leave-to {
  opacity: 0;
  max-width: 0;
  transform: translateX(-8px);
}
</style>
