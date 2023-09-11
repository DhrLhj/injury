

<template>

    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                <span style="font-size: 2.5vh;">等级1——炸伤照片</span>
                <div style="display: flex;height: 5%;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">照片</span>
                </div>
                <div style="width: 90%;height: 90%;">
                    <div style="height:100%">
                    <img :src="require('@/assets/imagebox/explosion/1.jpg')" class="carousel-image" alt="" style="height:100%;width:100%"> </div></div>
    
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
                            1、以止血带控制大血管破裂、创伤性截肢、大面积软组织撕脱等肢体致命性出血。<br>
                            2、止血带应扎于伤口近心端5-8cm处，紧急时可直接扎于出血部位近端衣裤表面。如应用皮带、布条、纱布条等作为临时止血带时，带宽不应＜5cm。<br>
                            3、迅速清理上呼吸道梗阻伤员口腔或鼻腔内血凝块、异物或分泌物，将伤员置于侧卧位或将头偏向一侧。<br>
                            4、针对能继续战斗的伤员可口服非甾体抗炎镇痛药，如美洛昔康（又称莫比可）片、对乙酰氨基酚（又称扑热息痛）片。<br><br>
    
                                            药品和工具：<br>
                                            止血带、美洛昔康片、对乙酰氨基酚片
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
    export default {
        name:"RightVideoSix",
        data(){
            return{   
                videourl:"/videos/炸伤救治.mp4",
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
//           this.jioaxue();
//           this.keysPressed = {};
//         } else if (this.keysPressed['a'] && this.keysPressed['f']) {
//           this.geren();
//           this.keysPressed = {};
//         } else if (this.keysPressed['a'] && this.keysPressed['e']) {
//           this.tongzhi();
//           this.keysPressed = {};
//         } else if (this.keysPressed['a'] && this.keysPressed['b']) {
//           this.shangqing();
//           this.keysPressed = {};
//         } else if (this.keysPressed['b'] && this.keysPressed['k']) {
//           this.jijiu();
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