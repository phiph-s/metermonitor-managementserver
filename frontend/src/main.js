import { createApp } from 'vue'
import AppWrapper from './AppWrapper.vue'
import router from "@/router";
import VueApexCharts from "vue3-apexcharts";

createApp(AppWrapper)
    .use(router)
    .use(VueApexCharts)
    .mount('#app')
