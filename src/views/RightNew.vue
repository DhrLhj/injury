<template>
    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 2%">
            <span style="font-size: 3vh;">如何辨别烧伤？</span>
            <div style="display: flex;font-size: 1.8vh;align-items: center;">
            <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
            <span style="margin-left: 0.5%;">视频</span>
            </div>
            <!-- <Vedio class="vedio" :vedioUrl="vedioUrl"/> -->
            <video ref="videoPlayer"  controls :src="videourl"></video>
            </div>
    </template>
    
    <script>
    export default {
        name:"Right"
    
    }
    </script>
    
    <style>
    
    </style>