<template>

  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <app></app>
  </n-config-provider>

</template>

<script setup>
import App from "@/App.vue";
import { NConfigProvider } from 'naive-ui';
import { darkTheme } from 'naive-ui';

const themeOverrides = {
    common: {
      primaryColor: '#18bcf2',
      bodyColor: '#111111',
    }
  }

</script>
<style>

</style>
