<template>
  <div style="width: 100%;height: 100%;">
    <div style="height: 100%;width: 100%;"> 
    <el-row :gutter="10"  style="height: 100%;margin-top: 3%;">
        <el-col :span="10" style="width: 100%;height: 100%;">
            <div style="font-size: 3vh;">
                总成绩
            </div>
            <div style="display: flex;
                flex-direction: column;
                justify-content: space-evenly;
                align-items: center;
                width: 100%;height: 60%;margin-top: 5%;border:2px solid #dcdfe6;border-radius: 10px;">
                <div>
                    <el-progress type="circle" :percentage="25" >
                        <template #default="{ percentage }">
                        <div class="circleText">
                            <span class="percentage-label">已学</span>
                        </div>
                        <div class="circleText" style="margin-top: 20%">
                            <span class="percentage-value">{{ percentage }}%</span>
                        </div>
                        </template>
                    </el-progress>
                </div>
                <div style="width: 100%;padding-left: 5%;padding-right: 5%;">
                    <span style="margin-left: 15%;">不及格</span>
                    <span style="margin-left: 20%;">良好</span>
                    <span style="margin-left: 25%;">优秀</span>
                    <el-steps :active="active" finish-status="success" style="width: 100%;">
                        <el-step title="0分" ></el-step>
                        <el-step title="60分"></el-step>
                        <el-step title="85分"/>
                        <el-step title="100分"/>
                    </el-steps>
                </div>
            </div>
                <div style="display: flex;margin-top:1%;font-size: 1.8vh;align-items: center;">
                    <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                    <span style="margin-left: 0.5%;">计分规则</span>
                </div>
                <p>最终成绩按照考核的百分制计分，满分为100分。</p>
                <p>满分100分=作业(30%)+考试(50%)+视频(20%)</p>
        </el-col>
        <el-col :span="14">
            <div style="font-size: 3vh;">
                成绩明细(满分100分)
            </div>
            <div style="display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
                height: 50%;
                margin-top: 4%;border:2px solid #dcdfe6;border-radius: 10px;">

    <el-collapse v-model="activeName" accordion style="width: 100%;height: 100%;">
      <el-collapse-item title="视频20分" name="1">
        <div>
          视频得分明细:  ...
        </div>
      </el-collapse-item>
      <el-collapse-item title="习题30分" name="2">
        <div>
            习题得分明细:  ...

        </div>
      </el-collapse-item>
      <el-collapse-item title="考试50分" name="3">
        <div>
            考试得分明细:  ...
        </div>
      </el-collapse-item>
    </el-collapse>

            </div>
        </el-col>
    </el-row>
    </div>
  </div>
</template>

<script>

export default {
    name:"TrainingResults",
    data(){
        return{
            active:4,
        }
    }
}
</script>

<style scoped>
.el-col{
    height: 100%;
}
.el-collapse-item{
    margin-left: 3%;
    margin-right: 3%;
    height: 20%;
    margin-top: 3%;
}
.el-collapse-item__header{
    font-size: 2vh;
    /* border-bottom-width: 3px; */
    border-bottom-color: black;
}
</style>