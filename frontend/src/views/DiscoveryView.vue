<template>
  <n-h2>Discovery</n-h2>

  Waiting for setup:

  <ul>
    <li v-for="item in discoveredMeters" :key="item.id">
      <router-link :to="'/setup/'+item">{{ item }}</router-link>
    </li>
  </ul>

  <n-h2>Watermeters:</n-h2>

  <ul>
    <li v-for="item in waterMeters" :key="item.id">
      <router-link :to="'/meter/'+item">{{ item }}</router-link>
    </li>
  </ul>
</template>

<script setup>
import {onMounted, ref} from 'vue';
import {NH2} from 'naive-ui';
import router from "@/router";

const discoveredMeters = ref([]);
const waterMeters = ref([]);


// add secret to header of fetch request
const getData = async () => {
  let response = await fetch(process.env.VUE_APP_HOST + '/api/discovery', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  discoveredMeters.value = await response.json();

  response = await fetch(process.env.VUE_APP_HOST + '/api/watermeters', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  waterMeters.value = await response.json();
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