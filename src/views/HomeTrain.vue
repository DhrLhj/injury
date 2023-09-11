<template>
  <div class="main">
    <div class="content">
      <!--        左侧图表    -->
      <div class="left-content">

        <div class="container">
          <div class="centered-text">伤情数据加载</div>
          <div class="buttons">

            <el-button style="width:100px; font-size: 20px; margin: 0 20px" size="medium" type="primary" @click="dialogVisible = true; initCamera()">拍摄</el-button>

            <el-dialog
                v-model="dialogVisible"
                title="Tips：请允许访问摄像头权限进行拍照"
                width="40%"
                @close = "stopNavigator, stopTracking"
                @open="startTracking"
            >
              <div style="width: 90%;margin-left: 10%;">

                <video id="videoCamera" style="width: 100%;padding: 3%;border-radius: 5%;border: 1px solid #036eff;" ref="videoElement" autoplay></video>



                <el-button type="primary" @click="takePhoto()" style="width: 100%;height: 20%;">拍照</el-button>
                <canvas style="width: 100%; height: 40vh;"
                        ref="canvasElement"></canvas>
                <!-- <img :src="photoUrl" v-if="photoUrl" alt="拍摄照片"> -->
              </div>
              <template #footer>
                <span class="dialog-footer">
                  <el-button @click="dialogVisible = false; stopNavigator()">取消</el-button>
                  <el-button type="primary" @click="dialogVisible = false; sendPhoto()">
                    提交
                  </el-button>
                </span>
              </template>
            </el-dialog>

            <el-upload ref="fileInput" action accept=".jpeg, .jpg, .png" :auto-upload="false" :show-file-list="false" :on-change="uploadFile">
              <el-button style="width:100px; font-size: 20px; margin: 0 20px" size="medium" type="primary" >载入</el-button>
            </el-upload>





          </div>
          <div class="centered-text">伤情数据记录</div>
          <div class="image-container" >
            <div class="slide" :class="{ 'selected': selectedImageIndex === index }" v-for="(image, index) in images" :key="index" >
              <img :src="image" alt="图片" class="bordered-image" @click="selectImage(index)" />
            </div>

          </div>
        </div>




      </div>
      <!--        右侧数据表-->
      <div class="right-content">
        <div class="centered-text">伤情实时识别</div>
        <div class="container" v-if="isPictrue">
            <div class="container1 left-column" >
              <img :src="images[this.selectedImageIndex]" alt="图片" class="bordered-image"/>
            </div>
            <div class="image-container right-column" >

              <div class="slide" v-for="(image, index) in images1" :key="index">
                <img :src="image" alt="图片" class="bordered-image"/>
              </div>

            </div>
        </div>
        <div class="container" v-else>
        </div>
        <div class="centered-text">救治方案推荐</div>

        <!-- <div>
          <button @click="speakText">播放语音</button>
        </div> -->
        <div v-if="isPictrue">


          <div id="box1" class="box" @click="clickBox1"  style="background-color: #99a9bf">
            <div style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px">识别结果</div>
            <div style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px">{{80}}%</div>
            <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="clickbutton(0)">{{88}}%  推荐方案1</el-button>
            <audio ref="audioElement" @canplaythrough="startPlayback">
              <source src="C:\Users\11985\Desktop\0905\admin-front\admin-front\public\MP3\盛夏-毛不易.320.mp3" type="audio/mpeg">您的浏览器不支持音频播放。
            </audio>
            <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="downloadData">{{8}}%  推荐方案2</el-button>
            <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="downloadData">{{4}}%  推荐方案3</el-button>





          </div>
          <div id="box2" class="box" @click="clickBox2" >
            <span style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px;">识别结果</span>
            <span style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px;">{{12}}%</span>

            <el-button style="width: 180px; font-size: 24px; margin: 0 40px;" size="big" type="primary" @click="downloadData">推荐方案</el-button>
          </div>



          <div id="box3" class="box" @click="clickBox3" >
            <span style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px">识别结果</span>
            <span style="width: 80px; font-family: bond; text-align: center; font-size: 20px; margin: 0 40px 0 40px">{{8}}%</span>


            <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="downloadData">推荐方案</el-button>
            <!-- <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="downloadData">推荐方案2</el-button>
            <el-button style="width:180px; font-size: 24px; margin: 0 40px 0 40px" size="big" type="primary" @click="downloadData">推荐方案3</el-button>-->

          </div>

        </div>

      </div>
    </div>
  </div>
</template>

<script>
import {ref} from "vue";
import {sendPhoto} from "@/api/api";
import router from '@/router/index.js';

export default {
  
  name:"homeTrain",
  data(){
    return{
      // url:"require('@/assets/imagebox/gunshot/1.jpg')"
      videourl:"/videos/骨折识别.mp4",
      videourl1:"././",
      videoStream: null,
      isPictrue: false,
      dialogVisible:ref(false),
      boxId: 1,
      images: [require('@/assets/leftimagebox/1.jpg'), require('@/assets/leftimagebox/2.jpg')],
      imagesId: 2 ,
      images1: [
        require('@/assets/leftimagebox/5.jpg'),require('@/assets/leftimagebox/6.jpg'),require('@/assets/leftimagebox/1.jpg')
      ],
      projectList:[["/study/right/1",'/study/right/burn','/study/right/burnDrug'],['/study/right/2','/study/right/fracture','/study/right/fractureDrug'],['/study/right/3','/study/right/bruise','/study/right/bruiseDrug'],
      ['/study/right/4','/study/right/scratches','/study/right/scratchesDrug'],['/study/right/5','/study/right/gunshot','/study/right/gunshotDrug'],['/study/right/6','/study/right/explosion','/study/right/explosionDrug']],
      selectedImageIndex: -1,
      output: '', // 初始化一个空字符串来存储输出的文本
      keysPressed: {},
    }
  },
  
  methods: {

    handleKeydown(event) {
        this.keysPressed[event.key] = true;
        if (this.keysPressed['c'] && this.keysPressed['b']) {//拍摄
        this.dialogVisible = true;
        this.initCamera()
        this.keysPressed = {};
        } else if (this.keysPressed['d'] && this.keysPressed['b']) {//拍照
          this.takePhoto();
          this.keysPressed = {};
        } else if (this.keysPressed['e'] && this.keysPressed['b']) {//提交
          this.dialogVisible = false;
          this.sendPhoto()
          this.keysPressed = {};
        } else if (this.keysPressed['f'] && this.keysPressed['b']) {//取消拍摄
          this.dialogVisible = false;
          this.stopNavigator();
          this.keysPressed = {};
        } else if (this.keysPressed['l'] && this.keysPressed['b']) {//方案选择1
          this.clickbutton(0);
          this.keysPressed = {};
        } else if (this.keysPressed['m'] && this.keysPressed['b']) {//方案选择2
          this.clickbutton(1);
          this.keysPressed = {};
        } else if (this.keysPressed['n'] && this.keysPressed['b']) {//方案选择3
          this.clickbutton(2);
          this.keysPressed = {};
        } else if (this.keysPressed['o'] && this.keysPressed['b']) {//2方案选择1
          this.clickbutton1(4);
          this.keysPressed = {};
        } else if (this.keysPressed['p'] && this.keysPressed['b']) {//3方案选择1
          this.clickbutton1(5);
          this.keysPressed = {};
        } else if (this.keysPressed['g'] && this.keysPressed['b']) {//载入
          this.uploadFile();
          this.keysPressed = {};
        } else if (this.keysPressed['h'] && this.keysPressed['b']) {//图片选择下滑
          this.selectedImageIndex += 1
          if (this.selectedImageIndex < this.images.length) {
            this.selectImage(this.selectedImageIndex);
          }else{
            this.selectedImageIndex = this.images.length - 1
          }
          this.keysPressed = {};
        } else if (this.keysPressed['i'] && this.keysPressed['b']) {//图片选择上滑
          this.selectedImageIndex -= 1
          if (this.selectedImageIndex > -1) {
            this.selectImage(this.selectedImageIndex);
          } else {
            this.selectedImageIndex = 0
          }
          this.keysPressed = {};
        } else if (this.keysPressed['j'] && this.keysPressed['b']) {//方案选择下滑
          this.boxId += 1
          if (this.boxId > 3) {
            this.boxId = 3
          }
          this.clickBox(this.boxId);
          this.keysPressed = {};
        } else if (this.keysPressed['k'] && this.keysPressed['b']) {//方案选择上滑
          this.boxId -= 1
          if (this.boxId < 1) {
            this.boxId = 1
          }
          this.clickBox(this.boxId);
          this.keysPressed = {};
        } else if (this.keysPressed['2'] && this.keysPressed['b']) {//语音播放
          this.speakText();
          console.log("aaa")
          this.keysPressed = {};
        } else if (this.keysPressed['r'] && this.keysPressed['b']) {//急救
          router.push("/study/right/detail");
          this.keysPressed = {};
        } else if (this.keysPressed['s'] && this.keysPressed['b']) {//教学
          router.push("/Teaching");
          this.keysPressed = {};
        }
    },

    handleKeyup(event) {
      delete this.keysPressed[event.key];
    },

    startTracking() {
      document.addEventListener('keydown', this.handleKeydown);
      document.addEventListener('keyup', this.handleKeyup);
    },
    stopTracking() {
      document.removeEventListener('keydown', this.handleKeydown);
      document.removeEventListener('keyup', this.handleKeyup);
    },
    selectImage(index) {
      this.selectedImageIndex = index;
      this.isPictrue = true;
      this.boxId = 1
      this.clickBox(this.boxId);
      console.log('选择了')

    },
    selOne() {
      console.log('选择了')
      // this.videourl='/videos/撞伤救治.mp4'
      // window.location.replace("/homeTrain")
      location.reload();
    },

    //语音
    playAudio() {
      const audioElement = this.$refs.audioElement;
      this.output = '1'
      // 在这里不直接播放，等待 canplaythrough 事件触发后再播放
    },
    startPlayback() {
      const audioElement = this.$refs.audioElement;
      audioElement.play();

    },
    pauseAudio() {
      const audioElement = this.$refs.audioElement;
      audioElement.pause();
    },
    speakText() {
      // 检查浏览器是否支持SpeechSynthesis API
      if ('speechSynthesis' in window) {
        const speechSynthesis = window.speechSynthesis;
        const textToSpeak = '你好，这是一个语音示例。'; // 要合成的文本

        // 创建一个SpeechSynthesisUtterance对象
        const utterance = new SpeechSynthesisUtterance(textToSpeak);

        // 使用默认的语音
        utterance.voice = speechSynthesis.getVoices()[0];

        // 开始语音合成
        speechSynthesis.speak(utterance);
      } else {
        alert('抱歉，你的浏览器不支持语音合成功能。');
      }
    },

    //方案推荐
    clickBox1() {
      this.boxId = 1;
      this.clickBox(this.boxId);
    },
    clickBox2() {
      this.boxId = 2;
      this.clickBox(this.boxId);
    },
    clickBox3() {
      this.boxId = 3;
      this.clickBox(this.boxId);
    },
    clickBox(boxId) {
      const box1 = document.getElementById('box1');
      const box2 = document.getElementById('box2');
      const box3 = document.getElementById('box3');
      if (box1 !== null){
        box1.style.backgroundColor = '#FFFFFF';
        box2.style.backgroundColor = '#FFFFFF';
        box3.style.backgroundColor = '#FFFFFF';
        if (boxId === 1) {
          box1.style.backgroundColor = '#99a9bf';
        } else if (boxId === 2) {
          box2.style.backgroundColor = '#99a9bf';
        } else if (boxId === 3) {
          box3.style.backgroundColor = '#99a9bf';
        }
      }
    },

    // initCamera() {
    //   navigator.mediaDevices.getUserMedia({video: true})
    //       .then(stream => {
    //         const videoElement = this.$refs.videoElement;
    //         videoElement.srcObject = stream;
    //         this.videoStream = stream;
    //
    //         videoElement.play();
    //       })
    //       .catch(error => {
    //         console.error('无法访问摄像头', error);
    //         this.$message.error("无法访问摄像头")
    //       });
    // },
    initCamera() {
      let nonDefaultCameraId = null;

      navigator.mediaDevices.enumerateDevices()
          .then(devices => {
              const videoDevices = devices.filter(device => device.kind === 'videoinput');
              
              if (videoDevices.length > 1) {
                  // 假设第一个摄像头是默认摄像头，我们选择第二个
                  nonDefaultCameraId = videoDevices[1].deviceId;

                  // 使用非默认摄像头
                  return navigator.mediaDevices.getUserMedia({ video: { deviceId: nonDefaultCameraId } });
              } else {
                  throw new Error("只有一个摄像头可用");
              }
          })
          .then(stream => {
              const videoElement = this.$refs.videoElement;
              videoElement.srcObject = stream;
              videoElement.play();
          })
          .catch(error => {
              console.error("错误:", error);
          });



      // const cameraId = '0'; // 选择1号相机
      // // navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: cameraId } } })
      // navigator.mediaDevices.getUserMedia({ video: true })
      //     .then(stream => {
      //       const videoElement = this.$refs.videoElement;
      //       videoElement.srcObject = stream;
      //       this.videoStream = stream;

      //       videoElement.play();
      //     })
      //     .catch(error => {
      //       console.error('无法访问摄像头', error);
      //       this.$message.error("无法访问摄像头")
      //     });
    },

    takePhoto() {
      const videoElement = this.$refs.videoElement;
      const canvasElement = this.$refs.canvasElement;
      const context = canvasElement.getContext('2d');

      // 将视频流的画面绘制到Canvas中
      context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    },
    sendPhoto() {
      const canvasElement = this.$refs.canvasElement;

      // 获取Canvas中的图像数据
      const imageData = canvasElement.toDataURL('image/png');
      this.images.push(imageData);
      this.stopNavigator();
      // // 显示拍摄的照片
      // this.photoUrl = imageData;

      //const file = imageData
      //const time = (new Date()).valueOf()
      //// const name = time + '.png'
      //const name = time
      //console.log("name"+name)
      //console.log("file+++"+file)
      //const conversions = this.base64ToFile(file, name)
      //const data = new FormData()
//
      //console.log()
      //data.append('file', conversions)
//
      //sendPhoto(data).then(res=>{
//
      //  if(res.code==='200'){
      //    console.log("成功了")
      //    this.$message.success("提交成功")
      //    this.value=true
      //    this.isTrue=false
//
      //  }else{
      //    this.$message.success("图片未能识别,请重新拍照提交")
      //    console.log("图片未能识别")
      //  }
      //})
    },
    base64ToFile(base64, fileName) {
      // 将base64按照 , 进行分割 将前缀  与后续内容分隔开
      const data = base64.split(',')
      // 利用正则表达式 从前缀中获取图片的类型信息（image/png、image/jpeg、image/webp等）
      const type = data[0].match(/:(.*?);/)[1]
      // 从图片的类型信息中 获取具体的文件格式后缀（png、jpeg、webp）
      const suffix = type.split('/')[1]
      // 使用atob()对base64数据进行解码  结果是一个文件数据流 以字符串的格式输出
      const bstr = window.atob(data[1])
      // 获取解码结果字符串的长度
      let n = bstr.length
      // 根据解码结果字符串的长度创建一个等长的整形数字数组
      // 但在创建时 所有元素初始值都为 0
      const u8arr = new Uint8Array(n)
      // 将整形数组的每个元素填充为解码结果字符串对应位置字符的UTF-16 编码单元
      while (n--) {
        // charCodeAt()：获取给定索引处字符对应的 UTF-16 代码单元
        u8arr[n] = bstr.charCodeAt(n)
      }
      // 利用构造函数创建File文件对象
      // new File(bits, name, options)
      const file = new File([u8arr], `${fileName}.${suffix}`, {
        type: type
      })
      // 将File文件对象返回给方法的调用者
      return file
    },
    stopNavigator() {
      if (this.videoStream) {
        const videoElement = this.$refs.videoElement;
        videoElement.srcObject = null;  // 关闭视频流
        const tracks = this.videoStream.getTracks();
        tracks.forEach(track => track.stop());
        this.videoStream = null;
        console.log("摄像头关闭了");
      } else {
        console.error('无法访问摄像头');
      }
    },

    uploadFile() {
      // const fileInput = event.target;
      // // 检查是否选择了文件
      // if (fileInput.files[0] === "") {
      if (this.imagesId < 3){
        this.images.push(require('@/assets/leftimagebox/3.jpg'));
        this.imagesId += 1;
        window.alert("已加载一张图片");
        setTimeout(() => {
          // 自动确认弹窗
          const alertBox = document.querySelector('.alert');
          console.log(alertBox)
          if (alertBox) {
            alertBox.click();
          }
        }, 2000); // 延迟2秒后自动确认
      }      
      // }else{
      // // 获取选择的文件
      // const file = fileInput.files[0];
      // // 检查文件类型
      // const fileURL = URL.createObjectURL(file);
      // this.images.push(fileURL);
      // fileInput.value = '';
      // }
    },

    clickbutton(bu_id){
      console.log(this.projectList[this.selectedImageIndex][bu_id])
      router.push(this.projectList[this.selectedImageIndex][bu_id] );
    },

    clickbutton1(bu_id){
      console.log(this.projectList[bu_id][0])
      router.push(this.projectList[bu_id][0]);
    }

  },
  created() {
    // console.log(this.$ws.readyState)
    // console.log(WebSocket.CLOSED)
    // if (this.$ws.readyState === WebSocket.CLOSED) { 
    //   console.log('ss')
    //   this.$ws = new WebSocket('ws://127.0.0.1:8000/room/123/'); }

    this.handleWebSocketMessage=(event)=> {
      // 处理 WebSocket 消息
      const message = event.data;
      if (message !== "-1"){
      console.log('WebSocket消息：', message);
      if (message === '0620') {//拍摄
        this.dialogVisible = true;
        this.initCamera()
      } else if (message === '1125') {//拍照
        this.takePhoto();
      } else if (message.includes('10000')) {//提交
        this.dialogVisible = false;
        this.sendPhoto()
      } else if (message === '0115') {//取消拍摄
        this.dialogVisible = false;
        this.stopNavigator();
      } else if (message==='1226') {//方案选择1
        this.clickbutton(0);
      } else if (message.includes('0120')) {//方案选择2
        this.clickbutton(1);
      } else if (message.includes('0121')) {//方案选择3
        this.clickbutton(2);
      } else if (message.includes('0615')) {//2方案选择1
        this.clickbutton1(4);
      } else if (message.includes('0715')) {//3方案选择1
        this.clickbutton1(5);
      } else if (message.includes('18')) {//载入
        this.uploadFile();
      } else if (message === '30') {//图片选择下滑
        this.selectedImageIndex += 1
        if (this.selectedImageIndex > this.images.length - 1) {
          this.selectedImageIndex = this.images.length - 1
        }else{
          this.boxId = 1;
          this.clickBox(this.boxId);
        }
        this.selectImage(this.selectedImageIndex);
      } else if (message === '29') {//图片选择上滑
        this.selectedImageIndex -= 1
        if (this.selectedImageIndex < 0) {
          this.selectedImageIndex = 0
        } else{
          this.boxId = 1;
          this.clickBox(this.boxId);
        }
        this.selectImage(this.selectedImageIndex);
      } else if (message === '15') {//方案选择下滑
        this.boxId += 1
        if (this.boxId > 3) {
          this.boxId = 3
        }
        this.clickBox(this.boxId);
      } else if (message === '26') {//方案选择上滑
        this.boxId -= 1
        if (this.boxId < 1) {
          this.boxId = 1
        }
        this.clickBox(this.boxId);
      } else if (message==='22') {//语音播放
        this.speakText();
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
    console.log("退出hometrain")
    // this.$ws.removeEventListener('message', this.handleWebSocketMessage);
    document.removeEventListener('keydown', this.handleKeydown);
      document.removeEventListener('keyup', this.handleKeyup);
  }
}
</script>

<style lang="scss">
.main {
  //min-width: 1536px;
  user-select: none;
  position: relative;

  .content {
    display: flex;

    .left-content {
      width: 15%;
      height: auto;
      margin: 20px 1.5% 60px 3.5%;
      // align-items: center;
      // justify-content: space-between;

      .container {
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      .centered-text {
        font-size: 30px;
        font-family: bond;
        text-align: center;
        margin: 20px 0 20px 0;
      }

      .button-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
      }
      .image-container {
        width: 100%;
        height: 600px; /* 高度自适应保持图片比例 */
        overflow: scroll;
        display: flex;
        flex-direction: column;
        box-sizing: border-box; /* 计算边框和内边距在容器尺寸内 */
        border: 1px solid #ddd; /* 外边框样式 */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 阴影样式 */
        padding: 10px; /* 内边距 */
      }



      .slider {
        display: flex;
        width: max-content;
      }

      .slide {
        flex-shrink: 0;
        margin-bottom: 10px;

        .bordered-image {
          border: 1px solid #000; /* 边框样式 */
          width: 100%; /* 宽度为容器的百分之百 */
          height: auto; /* 高度自适应保持图片比例 */
          box-sizing: border-box; /* 将边框计入图片尺寸内 */
        }
      }
      .selected {
        border: 2px solid red;
      }
      .buttons {
        display: flex;
        align-items: center;
        justify-content: space-between;

        .add_button {
          width: 35px;
          height: 35px;
          border: none;
          cursor: pointer;
          background-color: #eeeeee00;
        }

        .button-group {
          display: flex;
          margin-left: 20px;
        }

      }


    }

    .right-content {
      width: 75%;
      height: auto;
      margin: 20px 1.5% 60px 3.5%;
      align-items: center;
      justify-content: space-between;



      .centered-text {
        font-size: 30px;
        font-family: bond;
        margin: 20px 0 20px 20px;
      }

      .button-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
      }


      .container {
        width: 100%;
        height: 450px; /* 高度自适应保持图片比例 */
        overflow: hidden;

        display: flex;
        box-sizing: border-box; /* 计算边框和内边距在容器尺寸内 */
        border: 1px solid #ddd; /* 外边框样式 */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 阴影样式 */
        padding: 10px; /* 内边距 */
      }

      .container1 {
        width: 60%;
        height: 80%;
        margin: 0 5% 0 5%;

        display: flex;
        flex-direction: column;
        .bordered-image {

          width: auto;
          height: 420px;
          border: 1px solid #000; /* 边框样式 */
          box-sizing: border-box; /* 将边框计入图片尺寸内 */
        }
      }

      .image-container {
        width: 30%;
        height: 100%; /* 高度自适应保持图片比例 */

        display: flex;
        flex-direction: column;
        overflow: scroll;
        box-sizing: border-box; /* 计算边框和内边距在容器尺寸内 */
        border: 1px solid #ddd; /* 外边框样式 */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* 阴影样式 */
        padding: 10px; /* 内边距 */
      }

      .slider {
        display: flex;
        width: max-content;
      }

      .slide {
        flex-shrink: 0;
        margin-bottom: 10px;

        .bordered-image {
          border: 1px solid #000; /* 边框样式 */
          width: 100%; /* 宽度为容器的百分之百 */
          height: auto; /* 高度自适应保持图片比例 */
          box-sizing: border-box; /* 将边框计入图片尺寸内 */
        }












      }

      .box{
        margin: 0 0 15px 0;
        padding: 10px; /* 内边距 */
        background-color: #FFFFFF;
        display: flex;

      }
    }

  }
}

</style>
