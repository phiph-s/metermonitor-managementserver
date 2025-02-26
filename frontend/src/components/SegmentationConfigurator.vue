<template>
  <n-card>
    <template #cover>
      <img v-if="lastPicture" :src="'data:image/'+lastPicture.picture.format+';base64,' + lastPicture.picture.data" alt="Watermeter" />
    </template>
    <br>
    <label>
      <input type="number" v-model="nsegments" max="10" min="5" style="max-width: 32px"/>
      Segments
    </label><br>
    <label>
      <input type="checkbox" v-model="nextendedLastDigit"/>
      Extended last digit
    </label><br>
    <label>
      <input type="checkbox" v-model="nlast3DigitsNarrow"/>
      Last 3 digits are narrow
    </label>
    <template #action v-if="nencodedLatest">
      <n-flex justify="space-around" size="large">
        <img class="digit" v-for="base64 in JSON.parse(nencodedLatest)[0]" :src="'data:image/png;base64,' + base64" :key="base64" alt="D" style="max-width: 40px"/>
      </n-flex>
    </template>
  </n-card>
</template>

<script setup>
import {NCard, NFlex} from 'naive-ui';
import {defineProps, defineEmits, ref, watch} from 'vue';

const props = defineProps([
    'lastPicture',
    'segments',
    'extendedLastDigit',
    'last3DigitsNarrow',
    'encodedLatest'
]);
const emits = defineEmits(['update']);

watch(() => props.segments, (newVal) => {
  nsegments.value = newVal;
});
watch(() => props.extendedLastDigit, (newVal) => {
  nextendedLastDigit.value = newVal;
});
watch(() => props.last3DigitsNarrow, (newVal) => {
  nlast3DigitsNarrow.value = newVal;
});
watch(() => props.encodedLatest, (newVal) => {
  nencodedLatest.value = newVal;
});

const nsegments = ref(props.segments);
const nextendedLastDigit = ref(props.extendedLastDigit);
const nlast3DigitsNarrow = ref(props.last3DigitsNarrow);
const nencodedLatest = ref(props.encodedLatest);

watch([nsegments, nextendedLastDigit, nlast3DigitsNarrow], () => {
  emits('update', {
    segments: nsegments.value,
    extendedLastDigit: nextendedLastDigit.value,
    last3DigitsNarrow: nlast3DigitsNarrow.value
  });
});


</script>


<style scoped>

</style>