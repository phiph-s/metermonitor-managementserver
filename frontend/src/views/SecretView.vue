<template>
  <input type="text" v-model="secret" placeholder="Enter Secret">
  <button @click="saveSecret">Save</button>
</template>

<script setup>
import { ref } from 'vue';
import router from "@/router";

const saveSecret = () => {
  // to local storage
  localStorage.setItem('secret', secret.value);
  router.push('/');
}

const secret = ref('');
</script>

<style scoped>

</style>