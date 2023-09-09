import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import * as ElIcon from '@element-plus/icons-vue'
import 'element-plus/dist/index.css'
import './assets/css/global.css'
import App from './App.vue'
import router from './router'
import store from './store'
import request from './utils/request';
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import Viewer from 'v-viewer'
import WebSocketPlugin from './hooks/useRecordVideo'; // 假设你的 WebSocket 插件文件是这样命名的



const app = createApp(App)


app.use(ElementPlus, {
    locale: zhCn,
  })
Object.keys(ElIcon).forEach((key) => {app.component(key, ElIcon[key])
})
app.use(WebSocketPlugin).use(store).use(router).use(ElementPlus);
app.config.globalProperties.request=request;
app.mount('#app')
