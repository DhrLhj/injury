

<template>

    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                <span style="font-size: 2.5vh;">等级2——划伤照片</span>
                <div style="display: flex;height: 5%;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">照片</span>
                </div>
                <div style="width: 100%;height: 88%;">
                    <div style="height:100%">
                    <img :src="require('@/assets/imagebox/scratches/3.jpg')" class="carousel-image" alt="" style="height:100%;width:100%"> </div></div>
    
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
                            1、对于不适合使用肢体止血带的可压缩（外部）出血，或作为止血带移除的辅助手段，使用战斗纱布作为CoTCCC止血敷料的首选，直接应用于出血部位上方2-3英寸的皮肤。
    
                                            药品和工具：<br>
                                            肢体止血带、止血敷料、Xstat
                </el-text>
                </el-scrollbar>
            </div>
                <div style="display: flex;margin-top:1%;font-size: 1.8vh;align-items: center;">
                    <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                    <span style="margin-left: 0.5%;">救治操作视频</span>
                </div>
                    
            <div style="width: 100%;height:20%;margin-top: 1%;">
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
        name:"RightVideoOne",
        data(){
            return{   
                videourl:"/videos/划伤救治.mp4",
            }
        },
       
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