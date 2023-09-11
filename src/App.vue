<script setup>
import useRecord from './hooks/useRecord.js';
import useRecordVideo from './hooks/useRecordVideo';
import usekey from '@/hooks/usekey'

//通过ws传输音频(调用讯飞接口)
useRecord();
usekey();
//通过ws传输视频
// useRecordVideo();
</script>
<template>
  <div id="app">
    <router-view/>
  </div>
</template>
<style>
html,
body,
#app{
height: 100%;
}
</style>


