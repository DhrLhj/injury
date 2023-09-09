<template>

    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                <span style="font-size: 2.5vh;">等级3——烧伤照片</span>
                <div style="display: flex;height: 5%;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">照片</span>
                </div>
                <div style="width: 90%;height: 90%;">
                    <div style="height:100%">
                    <img :src="require('@/assets/imagebox/burn/4.jpg')" class="carousel-image" alt="" style="height:100%;width:100%"> </div></div>
    
    </div>
    <div style="display: inline-block;float: right;width: 100%;height: 100%;margin-top: 1%;padding-left: 2%;">
                
         <div style="width: 90%;height: 100%;margin-top: 1%;">
                   
            <div style="display: flex;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">救治方案</span>
            </div>
            <div style="padding:1%;border-radius:10px;border:2px solid #dcdfe6">
                <el-scrollbar height="40vh">
                <el-text size="default" tag="p">以下是一种可能的处理方案，但需要注意的是，实际处理应由专业的医疗人员进行，并根据士兵实际烧伤的严重程度和具体情况进行调整。<br>1、以冷水冲洗烧伤创面，冲洗后用烧伤敷料或干燥、清洁布单覆盖包裹创面；采取保温措施。<br>
                                            2、烧伤面积 20%的伤员可口服补液，推荐烧伤饮料（100ml含氯化钠0.3g,碳酸氢钠0.15g，苯巴比妥0.005g，加适量糖）。
                                            烧伤面积>20%的伤员应输入晶体液（乳酸钠林格液或等渗盐水）。体重40-80kg的成年人液体输入速度为X%（烧伤面积）×10ml/h。<br>
                                            3、对有面颈部烧伤、痰液有碳粒、声音嘶哑或失声、呼吸窘迫，听诊肺部有哮鸣音、干性啰音或捻发音等吸入性损伤征象的伤员，应密切观察其通气状况；如出现气道梗阻症状和体征，应紧急行环甲膜切开或气管插管，吸氧，维持SpO2>92%。<br>
                                            4、采取镇痛措施。<br><br>
    
                                            药品和工具：<br>
                                            烧伤敷料、烧伤饮料、晶体液、美洛昔康片、对乙酰氨基酚片
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
    import RightVideoOneVue from '../StudyVideo/RightVideoOne.vue';
    export default {
        name:"RightVideoOne",
        data(){
            return{   
                videourl:"/videos/烧伤救治.mp4",
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