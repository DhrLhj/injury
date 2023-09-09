<template>
  <video ref="videoElement" autoplay></video>
</template>

<script>

export default {
  data() {
    return {
      mediaStream: null,
      socket: null,
    };
  },
  mounted() {
    this.startCamera();
    this.setupWebSocket();
  },
  methods: {
    async startCamera() {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        this.$refs.videoElement.srcObject = this.mediaStream;
      } catch (error) {
        console.error('无法访问摄像头:', error);
      }
    },
    setupWebSocket() {
      // this.socket = io('ws://127.0.0.1:8000/room/123/'); // 替换为你的后端 WebSocket 服务器地址
      this.socket = new WebSocket('ws://127.0.0.1:8000/room/123/');
      this.socket.onopen = () => {
          console.log('WebSocket connection opened');
          
          // 在连接建立后，可以执行一些初始化操作
          // ...
        };
      
      // 发送视频流数据
      const videoElement = this.$refs.videoElement;
      videoElement.addEventListener('play', () => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        setInterval(() => {
          context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((blob) => {
            this.socket.send('video-stream', blob); // 发送视频流数据到后端
          }, 'video/webm');
        }, 1000 / 10); // 每秒 30 帧
      });
    },
  },
};
</script>

