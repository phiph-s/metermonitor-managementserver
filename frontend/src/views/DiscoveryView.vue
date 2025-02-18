<template>
  <n-h2>Waiting for setup</n-h2>
  <n-flex>
      <WaterMeterCard v-for="item in discoveredMeters" :key="item.id" :last_updated="item[1]" :meter_name="item[0]" :setup="true"/>
  </n-flex>

  <n-h2>Watermeters</n-h2>
  <n-flex>
      <WaterMeterCard v-for="item in waterMeters" :key="item.id" :last_updated="item[1]" :meter_name="item[0]" :setup="false"/>
  </n-flex>
</template>

<script setup>
import {onMounted, ref} from 'vue';
import {NH2, NFlex} from 'naive-ui';
import router from "@/router";
import WaterMeterCard from "@/components/WaterMeterCard.vue";

const discoveredMeters = ref([]);
const waterMeters = ref([]);


// add secret to header of fetch request
const getData = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + '/api/discovery', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  discoveredMeters.value = (await response.json())["watermeters"];

  response = await fetch(process.env.VUE_APP_HOST + '/api/watermeters', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  waterMeters.value = (await response.json())["watermeters"];
}

onMounted(() => {
  // check if secret is in local storage
  const secret = localStorage.getItem('secret');
  if (secret === null) {
    console.log(router)
    router.push({ path: '/unlock' });
  }

  getData();
});

</script>

<style scoped>

</style>