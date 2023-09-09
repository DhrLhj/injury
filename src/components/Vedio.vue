
<template>
      <video controls ref="videoPlayer" muted="muted" class="video-js video"></video>
</template>
<script setup>
import { onUnmounted, ref, nextTick, watch } from "vue"
import { useRoute, useRouter } from "vue-router";
//   import { LeftOutlined } from "@ant-design/icons-vue";
import videojs from "video.js"
import "video.js/dist/video-js.css"

const  props = defineProps({
  vedioUrl: {
    type: String
  }
});
const emits = defineEmits(["next"]);

watch(
    props.vedioUrl ,
    (newValue, oldValue)=> {
      console.log( newValue)
      //这里如果想要访问当前页面的一个函数的话必须保证函数在watch这段代码之前，否则会抱函数underfined。如果去掉immediate: true 那么函数位置随意。但是刚进入页面的时候这个函数是不执行的，你必须在onMount里面再去访问一次这个函数。
    },
    //可选immediate: true马上执行
    { deep: true, immediate: true },
);




const videoPlayer = ref(null)
const myPlayer = ref(null)
const route = useRoute();
const router = useRouter()
let token = route.query.token;


nextTick(() => {
    myPlayer.value = videojs(videoPlayer.value, {
      // poster: "//vjs.zencdn.net/v/oceans.png",//视频封面照片
      controls: true,//视频控件
      autoplay: true,//自动播放
      sources: [
        {
        //   src: `/api/video/playlist/${token}.m3u8`,//视频地址
          src:"/videos/枪弹伤救治.mp4",
          // src: props.vedioUrl,
          type: 'application/x-mpegURL',
        }
      ],
      controlBar: {
        remainingTimeDisplay: {
          displayNegative: false
        }
      },
      playbackRates: [0.5, 1, 1.5, 2]//设置播放速度
    },)
  })
  
  onUnmounted(() => {
    if (myPlayer.value) {
      myPlayer.value.dispose()
    }
})
</script>
<style lang="scss" scoped>
.video_wrap {
  width: 90vw;
  height: 100vh;
  position: relative;

    .backIndex {
      position: absolute;
      top: 0;
      left: 0;
      height: 50px;
      width: 100%;
      line-height: 50px;
      background: rgba(0, 0, 0, .5);
      z-index: 99;
      padding-left: 10px;
      font-size: 20px;
      font-weight: 400;
      opacity: 0;
      transition: all .3s;
      color: white;
  
      &:hover {
        opacity: 1;
      }
  
      span {
        cursor: pointer;
      }
    }
  
    .video {
      height: 100%;
      width: 100%;
    }
  
    ::v-deep(.vjs-big-play-button) {
      margin-left: 45%;
      margin-top: 20%;
    }
  }
  </style>
  
  
  
  