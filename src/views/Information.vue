<template>
  <div style="width: 100%;height: 100%;">
                <div style="margin:10vh 0 0 10vh;width:70% ;">
                <span>个人信息</span>
                <span style="float: right;color: #306eff;"><el-icon><Edit/></el-icon>编辑</span>
                <div style="height: 50vh;margin-top: 2vh;background-color: rgb(238, 239, 240);border-radius: 20px;">
                    
                    <el-row :gutter="20">
                        <el-col :span="2"><div class="grid-content ep-bg-purple" /></el-col>
                        <el-col :span="5"><div class="grid-content ep-bg-purple" />
                            <div style="margin-top: 40%;">
                                姓名：许三多
                            </div>
                            <div style="margin-top: 30%;">
                                所属战区：东部战区
                            </div>
                            <div style="margin-top: 30%;">
                                服役年限：3年
                            </div>
                        </el-col>
                        <el-col :span="8"><div class="grid-content ep-bg-purple" /></el-col>
                        <el-col :span="9"><div class="grid-content ep-bg-purple" />
                            <div style="margin-top: 20%;">
                                性别：男
                            </div>
                            <div style="margin-top: 15%;">
                                学历：无
                            </div>
                        </el-col>
                    </el-row>

                </div>
                </div>
            </div>
</template>

<script>

import router from '@/router/index.js';


export default {
    name:"Information",
    data(){
        return{
            keysPressed: {},
        }
    },
    methods: {
        handlenext() {
            document.querySelector("#TrainingResults").click();
        },
        handleKeydown(event) {
        this.keysPressed[event.key] = true;
        if (this.keysPressed['q'] && this.keysPressed['w']) {
          router.push("/home")
          this.keysPressed = {};
        }
      },
      handleKeyup(event) {
        delete this.keysPressed[event.key];
      },
        
    },
    created() {
      // if (this.$ws.readyState === WebSocket.CLOSED) { this.$ws = new WebSocket('ws://127.0.0.1:8000/room/123/'); }
      this.handleWebSocketMessage = (event) => {
        const message = event.data;
        if (message !== "-1"){
        console.log('WebSocket消息：', message);
        this.textmessage = 'WebSocket消息：' + message;

        if (message.includes('03')) {
            router.push("/home")
        } 
      }
      };
      this.$ws.addEventListener('message', this.handleWebSocketMessage);

    },
    mounted() {
        document.addEventListener('keydown', this.handleKeydown);
      document.addEventListener('keyup', this.handleKeyup);
    },
    beforeUnmount() {
      console.log("退出information")
      this.$ws.removeEventListener('message', this.handleWebSocketMessage);
      //this.$ws.close()
      document.removeEventListener('keydown', this.handleKeydown);
      document.removeEventListener('keyup', this.handleKeyup);
    }

}
</script>

<style>

</style>