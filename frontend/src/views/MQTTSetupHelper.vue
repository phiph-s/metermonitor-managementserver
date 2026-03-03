<script setup>
import {NDivider, NFlex} from 'naive-ui';

defineProps({
  config: {},
})
</script>

<template>
  <div>
    <n-flex gap="12px">
      <div>
        <b>Expected MQTT payload</b>
        <div class="code">
          {<br>
          "<b>name</b>": "unique name",<br>
          &nbsp;"WiFi-RSSI": -57 (optional),<br>
          &nbsp;"<b>picture</b>": {<br>
          &nbsp;&nbsp;"format": "jpeg" (optional),<br>
          &nbsp;&nbsp;"<b>timestamp</b>": "2025-..." or "1772383435",<br>
          &nbsp;&nbsp;"<b>data</b>": "iVBOR..." or "data:image/png;base64,iVBOR..."<br>
          &nbsp;}<br>
          }
        </div>
      </div>
      <n-divider vertical/>
      <div>
        <b>Broker</b><br/>
        Host: <span class="code-inline">{{ config?.mqtt?.broker }}</span><br/>
        Topic: <span class="code-inline">{{ config?.mqtt?.topic }}</span>
      </div>
    </n-flex>
  </div>
</template>

<style scoped>

.code {
  font-family: monospace, monospace;
  background-color: rgba(255, 255, 255, 0.1)
}

div.code {
  background: none;
}

.code-inline {
  font-family: monospace, monospace;
  padding: 2px 6px;
  border-radius: 6px;
  background-color: rgba(255, 255, 255, 0.08);
}
</style>