// GestureRecognition.vue

<template>
  <div>
    <!-- Vue 组件的模板部分 -->
    <button @click="startGestureRecognition" id="body">开始肢体动作识别</button>
    <!-- 显示来自后端的数据 -->
    <div>{{ gestureResult }}</div>
    <!-- Vue 组件的模板部分 -->
    <button @click="startGestureRecognition2" id="hand">开始手部模型识别</button>
    <!-- 显示来自后端的数据 -->
    <div>{{ gestureResult }}</div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      gestureResult: ''
    };
  },
  methods: {
    async startGestureRecognition() {
      try {
        // 向后端发送请求
        const response = await axios.get('http://localhost:5000/gesture-recognition');
        console.log(1); // 注意这里应该是console.log而不是print
        // 处理从后端返回的数据
        this.gestureResult = response.data;
      } catch (error) {
        console.error('发生错误:', error);
      }
    },
    async startGestureRecognition2() {
      try {
        // 向后端发送请求
        const response = await axios.get('http://localhost:5000/demo');
        console.log(1); // 注意这里应该是console.log而不是print
        // 处理从后端返回的数据
        this.gestureResult = response.data;
      } catch (error) {
        console.error('发生错误:', error);
      }
    }
  }
};
</script>