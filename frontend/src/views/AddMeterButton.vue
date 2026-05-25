<script setup>
import {AddOutlined, UsbFilled} from '@vicons/material';
import {HardwareChipOutline} from '@vicons/ionicons5';
import {NIcon, NFlex} from 'naive-ui';
import { defineEmits } from 'vue';

const emit = defineEmits([
    "click-add",
    "click-flash"
])
</script>

<template>
  <n-flex style="width: 375px; max-width: 100%;">
    <div class="add-card" data-testid="add-watermeter-card" style="" @click="emit('click-flash')">
      <div class="add-card-inner">
        <n-flex>
          <n-icon size="32">
            <UsbFilled/>
          </n-icon>
          <n-icon size="32">
            <HardwareChipOutline/>
          </n-icon>
        </n-flex>
        <span>Flash ESP</span>
      </div>
    </div>
    <div class="add-card" data-testid="add-watermeter-card" style="" @click="emit('click-add')">
      <div class="add-card-inner">
        <n-icon size="32">
          <AddOutlined/>
        </n-icon>
        <span>Add new</span>
      </div>
    </div>
  </n-flex>
</template>

<style scoped>

.add-card {
  width: calc(50% - 10px);
  height: 133px;
  min-height: 133px;
  border: 2px dashed rgba(255, 255, 255, 0.2);
  border-radius: 14px;
  margin-bottom: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  color: rgba(255, 255, 255, 0.7);
}

.add-card:hover {
  transform: translateY(-2px);
  border-color: rgba(255, 255, 255, 0.5);
  color: rgba(255, 255, 255, 0.9);
}

.light-mode .add-card {
  border-color: rgba(0, 0, 0, 0.2);
  color: rgba(0, 0, 0, 0.6);
}

.light-mode .add-card:hover {
  border-color: rgba(0, 0, 0, 0.4);
  color: rgba(0, 0, 0, 0.8);
}

.add-card-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 16px;
}

</style>