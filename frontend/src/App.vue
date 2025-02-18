<template>

  <n-space vertical size="large">
    <n-layout>
      <n-layout-header style="padding: 24px;">
        <n-gradient-text :size="24" type="danger">
          MeterMonitor
        </n-gradient-text>
      </n-layout-header>
      <n-layout-content content-style="padding: 24px;">
        <router-view></router-view>
      </n-layout-content>
    </n-layout>
  </n-space>

</template>

<script setup>
import {NLayout, NLayoutHeader, NLayoutContent, NSpace, NGradientText} from 'naive-ui';
</script>
<style>

</style>
