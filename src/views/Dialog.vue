<template>
  <!-- <el-button text @click="dialogVisible = true,initCamera()">
    点击拍照
  </el-button> -->
  <span class="msg" @click="dialogVisible = true,initCamera()"></span>
  <el-dialog
    v-model="dialogVisible"
    title="Tips：请允许访问摄像头权限进行拍照"
    width="40%"
  > 
    <div style="width: 80%;margin-left: 10%;">
    <video style="width: 100%;padding: 3%;border-radius: 5%;border: 1px solid #036eff;" ref="videoElement" autoplay></video>
      <el-button type="primary" @click="takePhoto" style="width: 100%;height: 20%;">拍照</el-button>
      <canvas style="width: 90%; height: 40vh;"
      ref="canvasElement"></canvas>
      <!-- <img :src="photoUrl" v-if="photoUrl" alt="拍摄照片"> -->
    </div>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="dialogVisible = false,sendPhoto()">
          提交
        </el-button>
      </span>
    </template>
  </el-dialog>


</template>

<script>
import { ref } from 'vue'
// import { ElMessageBox } from 'element-plus'
export default {
    name:"Dialog",
    data(){
        return{
            dialogVisible:ref(false),
            photoUrl: null,

        }
    },
    // mounted() {
    //   this.initCamera();
    // },
    methods:{
      initCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            const videoElement = this.$refs.videoElement;
            videoElement.srcObject = stream;
            videoElement.play();
          })
          .catch(error => {
            console.error('无法访问摄像头：', error);
          });
      },
      takePhoto() {
        const videoElement = this.$refs.videoElement;
        const canvasElement = this.$refs.canvasElement;
        const context = canvasElement.getContext('2d');
   
        // 将视频流的画面绘制到Canvas中
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
   
        // 获取Canvas中的图像数据
        const imageData = canvasElement.toDataURL('image/png');
   
        // 显示拍摄的照片
        this.photoUrl = imageData;
      },
      sendPhoto(){
        console.log("发送照片")
      },
      }
    }
</script>

<style>
.showDiv,.hiddenDiv{
    display: none;
}
</style>