<template>
    <div style="height: 100%;">
      <el-row :gutter="1" style="height:92%;margin-top: 1%;">
  
        <el-col :span="7" style="height: 100%;width: 100%;margin-left: 0.5%;">
          <router-link to="/homeTrain">
          <el-card @click="study" style="height: 100%;width: 100%;">
  
            <div class="image" 
              @mouseenter="showOverlayOne=true"
              @mouseleave="showOverlayOne=false" style="height: 100%;width: 100%;" >
            <img :src="require('@/assets/icon/home1.jpg')" id="img1" alt="" style="height: 100%;width: 100%;" />
            <div class="overlay" v-if="showOverlayOne">
              <span class="overlay-text">{{ overlayTextOne }}</span>
            </div> 
            </div>
          </el-card>
        </router-link>
        </el-col>
        <el-col :span="7" style="height: 100%;width: 100%;margin-left: 4.5%;">
          <router-link to="/study/right/detail">
          <el-card @click="examine" style="height: 100%;width:100%;">
            
            <div class="image"
              @mouseenter="showOverlayTwo=true"
              @mouseleave="showOverlayTwo=false" style="height: 100%;width: 100%;" >
            <img :src="require('@/assets/icon/home2.jpg')" alt="" style="height: 100%;width: 100%;" />
            <div class="overlay" v-if="showOverlayTwo">
              <span class="overlay-text">{{ overlayTextTwo }}</span>
            </div>  
          </div>
            
  
  
          </el-card>
        </router-link>
        </el-col>
  
        <el-col :span="7" style="height: 100%;width: 100%;margin-left: 4.5%;">
          <router-link to="/Teaching">
          <el-card @click="examine" style="height: 100%;width:100%;">
            <div class="image" 
              @mouseenter="showOverlayThree=true"
              @mouseleave="showOverlayThree=false" style="height: 100%;width: 100%;" >
            <img :src="require('@/assets/icon/home3.jpg')" alt="" style="height: 100%;width: 100%;" />
            <div class="overlay" v-if="showOverlayThree">
              <span class="overlay-text">{{ overlayTextThree }}</span>
            </div>  
          </div>
            
  
  
          </el-card>
        </router-link>
        </el-col>
      </el-row>
    </div>
    <div> {{ textmessage }} </div>
  </template>
  
<script>
import router from '@/router/index.js';

export default {
    name:"home",
    data(){
      return{
        showOverlayOne:false,
        showOverlayTwo:false,
        showOverlayThree:false,
        overlayTextOne:"伤情识别",
        overlayTextTwo:"急救指南",
        overlayTextThree:"教学模式",
        textmessage:"",
        keysPressed: {},
      }
    },
    methods:{
      study(){
        console.log("我发了个请求");

      },
      train(){
        console.log("训练模式");
        // $router.push('/train')
      }
      ,
      examine(){
        console.log("考核模式")
      },
      geren(){
        router.push("/Information");
      },
      tongzhi(){
        router.push("/notice");
      },
      shangqing(){
        router.push("/HomeTrain");
      },
      jijiu(){
        router.push("/study/right/detail");
      },
      jioaxue(){
        router.push("/Teaching");
      },
      shibie(){
        router.push("/body");
      },
      // removeListeners() {
      //       // Code to remove your event listeners
      //       // For example:
      //       this.$ws.removeEventListener('message', this.handleWebSocketMessage);
      //   },
      handleKeydown(event) {
        this.keysPressed[event.key] = true;
        console.log(this.keysPressed)
        // this.$set(this.keysPressed, event.key, true);
        if (this.keysPressed['a'] && this.keysPressed['d']) {
          this.jioaxue();
          this.keysPressed = {};
        } else if (this.keysPressed['a'] && this.keysPressed['f']) {
          this.geren();
          this.keysPressed = {};
        } else if (this.keysPressed['a'] && this.keysPressed['e']) {
          this.tongzhi();
          this.keysPressed = {};
        } else if (this.keysPressed['a'] && this.keysPressed['b']) {
          this.shangqing();
          this.keysPressed = {};
        } else if (this.keysPressed['b'] && this.keysPressed['k']) {
          this.jijiu();
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

        if (message.includes('22')) {
          this.geren();
        } else if (message.includes('15')) {
          this.tongzhi();
        } else if (message.includes('25')) {
          this.shangqing();
        } else if (message.includes('26')) {
          this.jijiu();
        } else if (message.includes('23')) {
          this.jioaxue();
        } else if (message.includes('0303')) {
          this.shibie();
        }
      }
      };
     // this.$ws.addEventListener('message', this.handleWebSocketMessage);

    },
    mounted() {
      document.addEventListener('keydown', this.handleKeydown);
      document.addEventListener('keyup', this.handleKeyup);
    },
    beforeUnmount() {
      console.log("退出home")
      this.$ws.removeEventListener('message', this.handleWebSocketMessage);
      document.removeEventListener('keydown', this.handleKeydown);
      document.removeEventListener('keyup', this.handleKeyup);
    }

}
  </script>
  
  <style>
  .el-card{
    margin-left: 3%;
    height: 100%;
    padding: 0;
    --el-card-padding:0px;
    cursor: pointer;
  }
  .el-col{
    height: 50%;
  }
  .el-card__body{

    border: none;
  }
  .el-row{
    margin-left: 0 !important;
    margin-right: 0 !important;
  }
  .img-one::before {
    content: "学习模式";
    letter-spacing:0.6vh;
    font-size: 4vh;
    display: flex;
    flex-flow: wrap;
    color: #ffffff;
    align-content: center;
    justify-content: center;
    position: absolute;
    bottom: 0;
    left: 0%;
    width: 100%;
    height: 32%;
    background-color: rgba(0,0,0,0.5);
  }
  .img-two::before {
    content: "训练模式";
    letter-spacing:0.6vh;
    font-size: 4vh;
    display: flex;
    flex-flow: wrap;
    color: #ffffff;
    align-content: center;
    justify-content: center;
    position: absolute;
    bottom: 0;
    left: 0%;
    width: 100%;
    height: 30%;
    background-color: rgba(0,0,0,0.5);
  }
  .img-three:hover::before {
    content: "考核模式";
    letter-spacing:0.6vh;
    font-size: 4vh;
    display: flex;
    flex-flow: wrap;
    color: #ffffff;
    align-content: center;
    justify-content: center;
    position: absolute;
    bottom: 0vh;
    left: 0;
    width: 100%;
    height: 30%;
    background-color: rgba(0,0,0,0.5);
  }
  .el-card__body{
    height: 100%;
    width: 100%;
  }
  
  
  
  .image-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 400px;
  }
  
  .image {
    position: relative;
    width: 300px;
    height: 300px;
    background-size: cover;
    background-position: center;
    cursor: pointer;
  }
  
  .overlay {
    position: absolute;
    bottom: 0;
    right: 0;
    width: 100%;
    height: 28%;
    background-color: rgba(0, 0, 0, 0.4);
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  .overlay-text {
    color: white;
    font-size: 1.5rem;
  }
  </style>