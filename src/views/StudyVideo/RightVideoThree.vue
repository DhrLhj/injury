<template>

    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                <span style="font-size: 2.5vh;">等级1——撞伤照片</span>
                <div style="display: flex;height: 5%;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">照片</span>
                </div>
                <div style="width: 90%;height: 90%;">
                    <div style="height:100%">
                    <img :src="require('@/assets/imagebox/bruise/1.jpg')" class="carousel-image" alt="" style="height:100%;width:100%"> </div></div>
    
    </div>
    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                
         <div style="width: 90%;height: 100%;margin-top: 1%;">
                   
            <div style="display: flex;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">救治方案</span>
            </div>
            <div style="padding:1%;border-radius:10px;border:2px solid #dcdfe6">
                <el-scrollbar height="40vh">
                <el-text size="default" tag="p">以下是一种可能的处理方案，但需要注意的是，实际处理应由专业的医疗人员进行，并根据士兵实际烧伤的严重程度和具体情况进行调整。<br>
                            1、搬运头、颈、背部撞击伤伤员时应固定头部和脊柱。<br>
                            2、对躯干撞击伤伤员容量复苏时应防止过量输液加重脏器损伤。<br>
                            3、脊髓伤神经源性休克容量复苏液体宜选择晶体液，复苏应达到SBPN110mmHg或MAP80mmHg。必要时，输入2000-3000ml液体后可应用去甲肾上腺素、多巴胺等血管活性药物调节血管张力。<br><br>
    
                                            药品和工具：<br>
                                            三角巾、骨盆固定带、制式夹板、晶体液、去甲肾上腺素、多巴胺
                </el-text>
                </el-scrollbar>
            </div>
                <div style="display: flex;margin-top:1%;font-size: 1.8vh;align-items: center;">
                    <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                    <span style="margin-left: 0.5%;">救治操作视频</span>
                </div>
                    
                <div style="width: 100%;height:100%;margin-top: 1%;">
                <video ref="videoPlayer" poster="@/assets/imagebox/videoPhoto.jpg" controls style="width: 100%;height: 80%;">
                    <source :src="videourl" type="video/mp4">
                </video>
            </div>
    
        </div>
    </div>
    
    </template>
    
    <script>
    import { ref } from 'vue';
    import RightVideoOneVue from '../StudyVideo/RightVideoOne.vue';
    export default {
        name:"RightVideoThree",
        data(){
            return{   
                videourl:"/videos/烧伤救治.mp4",
            }
        },
        methods:{
//     toggleFullScreen() {
//   const videoPlayer = this.$refs.videoPlayer;

//   // 检查浏览器是否支持全屏功能
//   if (document.fullscreenEnabled) {
//     // 如果不在全屏状态下，请求全屏权限
//     if (!document.fullscreenElement) {
//       videoPlayer.requestFullscreen().catch((error) => {
//         console.error('请求全屏权限失败：', error);
//       });
//     } else {
//       // 如果已经在全屏状态下，退出全屏
//       this.toExitFullscreen();
//     }
//   } else {
//     console.error('浏览器不支持全屏功能');
//   }
// },
//     toExitFullscreen() {
//       if (document.fullscreenElement) {
//         // 如果当前处于全屏状态，则退出全屏
//         if (document.exitFullscreen) {
//           document.exitFullscreen();
//         }
//       }
//     },
//     handleKeydown(event) {
//         this.keysPressed[event.key] = true;
//         console.log(this.keysPressed)
//         // this.$set(this.keysPressed, event.key, true);
//         if (this.keysPressed['a'] && this.keysPressed['d']) {
//           this.toggleFullScreen();
//           this.keysPressed = {};
//         } else if (this.keysPressed['a'] && this.keysPressed['f']) {
//           this.toExitFullscreen();
//           this.keysPressed = {};
//         }
//       },
//       handleKeyup(event) {
//         delete this.keysPressed[event.key];
//       },
  },
  created() {

    this.handleWebSocketMessage=(event)=> {
      // 处理 WebSocket 消息
      const message = event.data;
      if (message !== "-1"){
        // console.log('WebSocket消息：', message);
        // if (message === '25') {//全屏
        //   this.toggleFullScreen()
        // } else if (message === '19') {//退出全屏
        //   this.toExitFullscreen();
        // }

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
      // this.$ws.removeEventListener('message', this.handleWebSocketMessage);
      document.removeEventListener('keydown', this.handleKeydown);
      document.removeEventListener('keyup', this.handleKeyup);
    }
       
     }
    
    </script>
    
    <style>
    .el-carousel__container{
        height: 100%;
    }
    
    .el-carousel__item h3 {
      color: #475669;
      opacity: 0.75;
      line-height: 300px;
      margin: 0;
      text-align: center;
    }
    
    .el-carousel__item:nth-child(2n) {
      background-color: #99a9bf;
    }
    
    .el-carousel__item:nth-child(2n + 1) {
      background-color: #d3dce6;
    }
    .el-carousel{
        height: 90%;
    }
    .el-carousel__container{
        height: 100%;
    }
    .scrollbar-demo-item {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 50px;
      margin: 10px;
      text-align: center;
      border-radius: 4px;
      background: var(--el-color-primary-light-9);
      color: var(--el-color-primary);
    }
    p{
        line-height: 4vh;
    }
    
    </style>