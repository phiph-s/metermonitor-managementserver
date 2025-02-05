<template>
  <h2>Setup for {{ id }}</h2>

  <img :src="'data:image/'+info.picture.format+';base64,' + info.picture.data" alt="Watermeter"/>
  {{ info2 }}
</template>

<script setup>
import {onMounted, ref} from 'vue';
import router from "@/router";
import { useRoute } from 'vue-router';

const route = useRoute();
const id = route.params.id;

const info = ref("");
const info2 = ref("");
const info3 = ref("");

const getData = async () => {
  let response = await fetch('/api/watermeters/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  info.value = await response.json();

  response = await fetch('/api/watermeters/' + id + '/evals', {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  info2.value = await response.json();

  response = await fetch('/api/thresholds/' + id, {
    headers: {
      'secret': `${localStorage.getItem('secret')}`
    }
  });
  info3.value = await response.json();

}

onMounted(() => {
  // check if secret is in local storage
  console.log(route)
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