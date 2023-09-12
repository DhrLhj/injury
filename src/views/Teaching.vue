<template>
  <div style="height: 100%;">
    <el-row :gutter="1" style="height:52%;margin-top: 0.8%;">

      <el-col :span="7" style="height: 100%;width: 100%;margin-left: 0.5%;">
        <router-link to="/study/right/1">
        <el-card @click="study" style="height: 100%;width: 100%;">

          <div class="image" 
            @mouseenter="showOverlayOne=true"
            @mouseleave="showOverlayOne=false" style="height: 100%;width: 100%;" >
          <img :src="require('@/assets/icon/xuexi.jpg')" id="img1" alt="" style="height: 100%;width: 100%;" />
          <div class="overlay" v-if="showOverlayOne">
            <span class="overlay-text">{{ overlayTextOne }}</span>
          </div> 
          </div>
        </el-card>
      </router-link>
      </el-col>


      <el-col :span="7" style="height: 100%;width: 100%;margin-left: 4.5%;">
        <router-link to="/train">
        <el-card @click="examine" style="height: 100%;width:100%;">
          
          <div class="image"
            @mouseenter="showOverlayTwo=true"
            @mouseleave="showOverlayTwo=false" style="height: 100%;width: 100%;" >
          <img :src="require('@/assets/icon/xunlian.jpg')" alt="" style="height: 100%;width: 100%;" />
          <div class="overlay" v-if="showOverlayTwo">
            <span class="overlay-text">{{ overlayTextTwo }}</span>
          </div>  
        </div>
          


        </el-card>
      </router-link>
      </el-col>

      <el-col :span="7" style="height: 100%;width: 100%;margin-left: 4.5%;">
        <router-link to="/examine">
        <el-card @click="examine" style="height: 100%;width:100%;">
          <div class="image" 
            @mouseenter="showOverlayThree=true"
            @mouseleave="showOverlayThree=false" style="height: 100%;width: 100%;" >
          <img :src="require('@/assets/icon/kaohe.jpg')" alt="" style="height: 100%;width: 100%;" />
          <div class="overlay" v-if="showOverlayThree">
            <span class="overlay-text">{{ overlayTextThree }}</span>
          </div>  
        </div>
          


        </el-card>
      </router-link>
      </el-col>






      <b style="margin-left: 2%;font-size: 3vh;color: #036eff;">热门易错伤情</b>
    </el-row>
    <el-row style="margin-top: 2%;margin-left: 2%;height: 40%;">
      <el-col :span="7" style="height: 100%;width: 100%;">
        <el-card style="height: 100%;margin-left: 5%;">
          <div style="height: 100%;margin-top: -7px;">
          <el-image :src="require('@/assets/icon/home4.jpg')" alt="" style="height: 30vh;width: 100%;position: relative;" />
          <div style="color: #036eff;margin-left: 2%;margin-right: 2%;font-size: 2.5vh;">
          <span>烧伤</span>
          <span style="float: right;">识别错误数：1000+</span>
          </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="7" style="height: 100%;width: 100%;margin-left: 5%;">
        <el-card style="height: 100%;">
          <div style="height: 100%;margin-top: -7px;">
          <el-image :src="require('@/assets/icon/home5.jpg')" alt="" style="height: 30vh;width: 100%;" />
          <div style="color: #036eff;margin-left: 2%;margin-right: 2%;font-size: 2.5vh;">
          <span>炸伤</span>
          <span style="float: right;">识别错误数：1000+</span>
          </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="7" style="height: 100%;width: 100%;margin-left: 5%;">
        <el-card style="height: 100%;">
          <div style="height: 100%;margin-top: -7px;">
          <el-image :src="require('@/assets/icon/home6.jpg')" alt="" style="height: 30vh;width: 100%;" />
          <div style="color: #036eff;margin-left: 2%;margin-right: 2%;font-size: 2.5vh;">
          <span>撞伤</span>
          <span style="float: right;">识别错误数：1000+</span>
          </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import router from "@/router";

export default {
    name:"Teaching",
    data(){
      return{
        showOverlayOne:false,
        showOverlayTwo:false,
        showOverlayThree:false,
        overlayTextOne:"学习模式",
        overlayTextTwo:"训练模式",
        overlayTextThree:"考核模式",
        keysPressed: {},
      }
    

    },
    methods:{
      study(){
        console.log("学习模式");
      },
      train(){
        console.log("问答模式");
        // $router.push('/train')
      }
      ,
      examine(){
        console.log("考核模式")
      },
      hh_hh(){
        router.push("/studyNew");
      },ii_ii(){
        router.push("/train");
      },bad_bad(){
        router.push("/examine");
      },

      handleKeydown(event) {
        this.keysPressed[event.key] = true;
        if (this.keysPressed['c'] && this.keysPressed['d']) {
          this.hh_hh();
          this.keysPressed = {};
        } else if (this.keysPressed['c'] && this.keysPressed['e']) {
          this.ii_ii();
          this.keysPressed = {};
        } else if (this.keysPressed['c'] && this.keysPressed['f']) {
          this.bad_bad();
          this.keysPressed = {};
        } else if (this.keysPressed['c'] && this.keysPressed['h']) {//急救
          router.push("/HomeTrain");
          this.keysPressed = {};
        } else if (this.keysPressed['c'] && this.keysPressed['g']) {//教学
          router.push("/Teaching");
          this.keysPressed = {};
        }
      },
      handleKeyup(event) {
        delete this.keysPressed[event.key];
      },
    },
    created() 
    {

      this.handleWebSocketMessage = (event) => {
        const message = event.data;
        if (message !== "-1"){
          console.log('WebSocket消息：', message);
          this.textmessage = 'WebSocket消息：' + message;

          if (message === '0515') { //学习
            this.hh_hh();
          } else if (message === '0520') { //训练
            this.ii_ii();
          } else if (message === '0521') { //考核
            this.bad_bad();
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
      console.log("退出teach")
      // this.$ws.removeEventListener('message', this.handleWebSocketMessage);
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
  content: "问答模式";
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