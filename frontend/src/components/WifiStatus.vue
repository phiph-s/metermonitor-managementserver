<template>
  <n-tooltip>
    <template #trigger>
      <n-tag :type="tagType" round>
        <NIcon>
          <template #default>
            <component :is="iconComponent" />
          </template>
        </NIcon>
        {{ signalLabel }}
      </n-tag>
    </template>
    <span>
      {{rssi}} dBm
    </span>
  </n-tooltip>
</template>

<script setup>
import { computed, defineProps } from 'vue'
import { NTag, NIcon, NTooltip } from 'naive-ui'

// Import Material Twotone WiFi icons
import {
  NetworkWifi1BarTwotone,
  NetworkWifi2BarTwotone,
  NetworkWifi3BarTwotone,
  NetworkWifiTwotone,
  SignalWifi0BarFilled
} from '@vicons/material'

const props = defineProps({
  rssi: {
    type: Number,
    required: true
  }
})

const signalLabel = computed(() => {
  const rssi = props.rssi
  if (rssi >= -50) return 'Excellent'
  if (rssi >= -60) return 'Very Good'
  if (rssi >= -70) return 'Good'
  if (rssi >= -80) return 'Fair'
  if (rssi >= -90) return 'Weak'
  return 'No Signal'
})

const tagType = computed(() => {
  const rssi = props.rssi
  if (rssi >= -60) return 'success'
  if (rssi >= -70) return 'warning'
  if (rssi >= -90) return 'error'
  return 'error'
})

const iconComponent = computed(() => {
  const rssi = props.rssi
  if (rssi >= -50) return NetworkWifiTwotone
  if (rssi >= -60) return NetworkWifi3BarTwotone
  if (rssi >= -70) return NetworkWifi2BarTwotone
  if (rssi >= -80) return NetworkWifi1BarTwotone
  if (rssi >= -90) return NetworkWifi1BarTwotone
  return SignalWifi0BarFilled
})
</script>
