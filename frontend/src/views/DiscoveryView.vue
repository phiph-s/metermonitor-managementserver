<template>
  <h2>Discovery</h2>

  Waiting for setup:

  <ul>
    <li v-for="item in waterMeters" :key="item.id">
      <router-link :to="'/setup/'+item">{{ item }}</router-link>
    </li>
  </ul>
</template>

<script setup>
import {onMounted, ref} from 'vue';
import router from "@/router";

const waterMeters = ref([]);

// add secret to header of fetch request
const getData = async () => {
  const response = await fetch('/api/discovery', {
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