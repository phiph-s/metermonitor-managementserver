<template>

  <n-space vertical size="large">
    <n-layout>
      <n-layout-content content-style="padding: 24px;">
        <div class="app-header">
          <div class="header-left">
            <transition name="back-slide" mode="out-in">
              <router-link v-if="showBack" to="/" key="back">
                <n-button quaternary round size="large" style="padding: 0; font-size: 16px;">
                  ← Back
                </n-button>
              </router-link>
            </transition>
            <img
              src="@/assets/logo.png"
              alt="Logo"
              class="theme-revert header-logo"
              :class="{ 'no-back': !showBack }"
            />
            <n-tag size="small" :type="buildTagType" class="build-tag">
              {{ buildTagLabel }}
            </n-tag>
          </div>
          <div class="header-right">
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
              :loading="headerState.refreshLoading"
              @click="headerState.onRefresh && headerState.onRefresh()"
              round
              size="large"
            >
              Refresh
            </n-button>
          </div>
        </div>
        <router-view></router-view>
      </n-layout-content>
    </n-layout>
  </n-space>

</template>

<script setup>
import {NLayout, NLayoutContent, NSpace, NButton, NIcon, NTooltip, NTag, useNotification} from 'naive-ui';
import { LightModeOutlined, DarkModeOutlined } from '@vicons/material';
import {onMounted, onUnmounted, ref, computed, reactive, provide} from "vue";
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

const alerts = ref([]);

const notification = useNotification();
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

const updateAlerts = async () => {
  const r = await fetch(host + 'api/alerts', {
    headers: {secret: localStorage.getItem('secret')}
  })

  if (r.status === 401) {
    await router.push({path: '/unlock'});
    return;
  }

  alerts.value = await r.json();
  notification.destroyAll();
  for (const alert of Object.keys(alerts.value)) {
    notification.create({
      title: alert.toUpperCase(),
      content: alerts.value[alert],
      closable: false,
      type: 'error'
    });
  }
}
const interval = ref(null);
onMounted(() => {
  websocketStopped.value = false;
  updateAlerts();
  interval.value = setInterval(updateAlerts, 60000);
  connectWebsocket();
});

onUnmounted(() => {
  websocketStopped.value = true;
  clearInterval(interval.value);
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
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.header-logo {
  max-width: 100px;
  margin-left: 20px;
  transition: margin-left 0.2s ease;
}

.n-dialog{
  border-radius: 12px;
}

.header-logo.no-back {
  margin-left: 0;
}

.build-tag {
  margin-left: 4px;
}

.back-slide-enter-active,
.back-slide-leave-active {
  transition: all 0.2s ease;
}

.back-slide-enter-from {
  opacity: 0;
  transform: translateX(-8px);
}

.back-slide-leave-to {
  opacity: 0;
  transform: translateX(-8px);
}
</style>
