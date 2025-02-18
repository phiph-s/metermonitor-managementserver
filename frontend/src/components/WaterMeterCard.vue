
<template>
  <n-card :title="meter_name" size="small" style="max-width: 300px;">
    <n-flex justify="space-between">
      <span>Last updated: {{ last_updated_locale }}</span>
    </n-flex>
    <template #action>
      <n-flex justify="end">
        <router-link v-if="setup" :to="'/setup/'+meter_name"><n-button type="primary">Setup</n-button></router-link>
        <router-link v-else :to="'/meter/'+meter_name"><n-button type="primary">View</n-button></router-link>
      </n-flex>
    </template>
  </n-card>
</template>

<script setup>
import {NCard, NButton, NFlex} from 'naive-ui';
import {defineProps} from 'vue';

const props = defineProps([
    'meter_name',
    'last_updated', // eg "2025-02-04T03:15:31"
    'setup'
]);

const last_updated_locale = new Date(props.last_updated).toLocaleString();
</script>

<style scoped>

</style>