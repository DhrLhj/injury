<template>
  <div class="main">
    <div class="buttons">
      <el-button class="button" size="big" type="primary" @click="downLoads">下载</el-button>
    </div>
    <div class="lines"></div>
    <button @click="handleZoom">放大</button>
    <viewer :images="descImgs" class="descimgBox">
      <img v-for="(decImg, index) in descImgs" :key="index" :src="decImg">
    </viewer>

  </div>




</template>

<script>
import { createApp } from 'vue';
import Viewer from 'v-viewer';
import 'viewerjs/dist/viewer.css';


const app = createApp(App);
app.use(Viewer);
Viewer.setDefaults({Options: { 'inline': true, 'button': true, 'navbar': true, 'title': true, 'toolbar': true, 'tooltip': true, 'movable': true, 'zoomable': true, 'rotatable': true, 'scalable': true, 'transition': true, 'fullscreen': true, 'keyboard': true, 'url': 'data-source' }
})

export default {
  name: "change",
  data() {
    return {
      downloadURL: '',
      descImgs: [
        'https://tse1-mm.cn.bing.net/th?id=OIP.fLI-fIeiAEMZwLhz6KkcMQAAA&w=201&h=200&c=7&o=5&pid=1.7',
        'https://tse1-mm.cn.bing.net/th?id=OIP.fLI-fIeiAEMZwLhz6KkcMQAAA&w=201&h=200&c=7&o=5&pid=1.7',
        'https://tse1-mm.cn.bing.net/th?id=OIP.fLI-fIeiAEMZwLhz6KkcMQAAA&w=201&h=200&c=7&o=5&pid=1.7',
        'https://tse1-mm.cn.bing.net/th?id=OIP.fLI-fIeiAEMZwLhz6KkcMQAAA&w=201&h=200&c=7&o=5&pid=1.7',
        'https://tse1-mm.cn.bing.net/th?id=OIP.fLI-fIeiAEMZwLhz6KkcMQAAA&w=201&h=200&c=7&o=5&pid=1.7'
      ]
    }
  },

  methods: {
    handleZoom() {
      // 触发自定义事件
      this.$emit('zoom');
    },
    downLoads(){
      let params = {}

      this.$http.getDownLoads(params).then((res) => {
        if (res) {
          console.log(res)

          const link = document.createElement('a');
          this.downloadURL = Global.DestURL.baseUrl + res.data
          link.href = this.downloadURL;
          console.log(this.downloadURL)

          link.download = 'zr_sim.zip'; // 指定下载文件名
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      })
        .catch((error) => {
          console.error(error)
        })
    }
  },
  created() {
    // const ws = new WebSocket('ws://127.0.0.1:8000/room/123/'); // 替换成实际的 WebSocket URL
    // ws.onmessage = (e) => {
    //   const message = e.data;
    //   if (message.includes("向下")) {
    //     this.handleZoom(); // 触发 information 界面的 change 事件
    //   }
    // }
  },
  mounted() {
    // 初始化 Viewer.js 实例
    // this.viewer = new Viewer(this.$refs.viewerContainer);
    // // 注册自定义事件
    // this.$on('zoom', () => {
    //   this.viewer.zoom(1); // 调用 Viewer.js 的 zoom 方法，将 zoom 参数设置为 1 表示放大
    // });
  },
}
</script>

<style lang="scss">
.main {
  //min-width: 1536px;
  user-select: none;
  position: relative;




  .lines {
    //height: calc(100vh * 540 / 950);
    height: 500px;
    display: flex;
    flex-direction: column;
    //margin-top: calc(100vh * 1 / 950);
    margin: 40px 300px 40px 300px;

    .line {
      display: flex;
      //margin-top: calc(100vh * 20 / 950);
      margin-top: 20px;
      //max-height: calc(100vh * 104 / 950);
      max-height: 104px;
      flex-grow: 1;


      .line-left {
        display: flex;
        justify-content: center;
        align-items: center;
        //width: calc(100vw * 200 / 1920);
        width: 200px;
        //max-height: calc(100vh * 104 / 950);
        max-height: 104px;
        height: 100%;
        background: #F3F3F3;
        border-radius: 2px 2px 2px 2px;
        border: 1px solid #E8E8E8;
        text-align: center;

        .line-text {
          font-size: 16px;
          font-weight: bold;
          color: #333333;
        }
      }

      .line-right {
        display: flex;
        overflow: auto;
        justify-content: center;
        align-items: center;

        //width: calc(100vw * 556 / 1920);
        width: 1572px;
        //max-height: calc(100vh * 104 / 950);

        max-height: 104px;
        height: 100%;
        background: #FAFAFA;
        border-radius: 2px 2px 2px 2px;
        border: 1px solid #F3F3F3;
        text-align: center;

        .line-text {
          width: 1000px;
          max-height: 104px;
          height: 100px;
          background: #FAFAFA;
          border-radius: 2px 2px 2px 2px;
          border: 1px solid #FAFAFA;
          text-align: center;
          font-size: 16px;
          font-weight: bold;
          color: #333333;
        }
      }
    }
  }

  .top2 {
    display: flex;
    align-items: center;
    justify-content: center;
    //margin: calc(100vw * 40 / 1920) auto 0 auto;
    margin: 40px auto 0 auto;
    //padding-bottom: calc(100vw * 40 / 1920);

    padding-bottom: 40px;
    text-align: center;
    border-bottom: 1px solid #E4EFF8;

    .card {
      //width: calc(100vw * 239 / 1920);
      width: 239px;
      background: #FFFFFF;
      border: 1px solid #E4EFF8;
      margin-right: 20px;
      //padding: calc(100vh * 16 / 950) calc(100vw * 23 / 1920) calc(100vh * 50 / 950) calc(100vw * 23 / 1920);
      padding: 25px 23px 25px 23px;
      cursor: pointer;
      transition: .3s;

      &:hover {
        transition: .3s;
        transform: scale(1.05);
      }




      .title {
        font-size: 18px;
        color: #253554;
      }
    }

    .card2 {
      //width: calc(100vw * 239 / 1920);
      width: 239px;
      background: #FFFFFF;
      margin-right: 20px;
      //padding: calc(100vh * 16 / 950) calc(100vw * 23 / 1920) calc(100vh * 50 / 950) calc(100vw * 23 / 1920);
      padding: 25px 23px 25px 23px;
      cursor: pointer;
      transition: .3s;

      &:hover {
        transition: .3s;
        transform: scale(1.05);
      }
    }
  }
  .buttons {
    display: flex;
    margin-top: 40px;
    margin-right: 200px;
    justify-content: right;
    align-items: center;

    button {
      width: 200px;
      margin: 0 16px;
    }
  }
}
</style>
