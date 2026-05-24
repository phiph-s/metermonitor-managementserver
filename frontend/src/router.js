import { createRouter, createWebHashHistory } from 'vue-router';

// Import your components for the routes
import DiscoveryView from "@/views/DiscoveryView.vue";
import SecretView from "@/views/SecretView.vue";
import SetupView from "@/views/SetupView.vue";
import MeterView from "@/views/MeterView.vue";
import SettingsView from "@/views/SettingsView.vue";
import FlashView from "@/views/FlashView.vue";

const routes = [
  {
    path: '/',
    name: 'Discovery',
    component: DiscoveryView
  },
  {
    path: '/flash',
    name: 'Flash',
    component: FlashView
  },
  {
    path: '/settings',
    name: 'Settings',
    component: SettingsView
  },
  {
    path: '/list',
    name: 'List',
    component: DiscoveryView
  },
  {
    path: '/unlock',
    name: 'Unlock',
    component: SecretView
  },
  {
    path: '/setup/:id',
    name: 'Setup',
    component: SetupView
  },
  {
    path: '/meter/:id',
    name: 'Meter',
    component: MeterView
  }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes
});

export default router;
