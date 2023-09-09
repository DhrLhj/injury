import { createStore } from 'vuex'

export default createStore({
  state: {
    permissions: false,
        url: '',
        //ws参数
        path: '',
        ws: null,//建立的连接
        lockReconnect: false,//是否真正建立连接
        timeout: 58*1000,//58秒一次心跳
        timeoutObj: null,//心跳心跳倒计时
        serverTimeoutObj: null,//心跳倒计时
        timeoutnum: null,//断开 重连倒计时
  },
  getters: {
  },
  mutations: {
  },
  actions: {
  },
  modules: {
  }
})
