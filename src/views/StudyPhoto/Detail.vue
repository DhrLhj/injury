<!-- <template>
        <div style="margin-top: 3%;height: 94%;margin-right: 3%;border: 2px solid #dcdfe6;
        border-radius: 10px;display: flex;">
            <el-row :gutter="10"  style="height: 100%;width: 100%;">
                <el-col :span="8">
                    <div class="grid-content ep-bg-purple" >
                     <div style="display: flex;flex-direction: column;align-items: center">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;margin-bottom: 3%;">烧伤</span>
                    </div>
                    <div style="display: flex;flex-direction: column;align-items: center;">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;margin-bottom: 3%;">撞伤</span>
                    </div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="grid-content ep-bg-purple" >
                    <div style="display: flex;flex-direction: column;align-items: center">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;;margin-bottom: 3%;">骨折</span>
                    </div> 
                    <div style="display: flex;flex-direction: column;align-items: center">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;;margin-bottom: 3%;">划伤</span>
                    </div>
                    </div>
                </el-col>
                <el-col :span="8">
                    <div class="grid-content ep-bg-purple" >
                     <div style="display: flex;flex-direction: column;align-items: center">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;margin-bottom: 3%;">烧伤</span>
                    </div>
                    <div style="display: flex;flex-direction: column;align-items: center">
                    <el-progress type="circle" :percentage="25" />
                    <span style="margin-top: 3%;margin-bottom: 3%;">枪弹伤</span>
                    </div>
                    </div>
                </el-col>
            </el-row>
        </div>
</template> -->
<template>
    <div style="margin-top: 3%;height: 94%;border: 2px solid #dcdfe6;
    border-radius: 10px;display: flex;">
        <el-row :gutter="10"  style="height: 100%;width: 100%;align-items: center;">
            <el-col :span="8">
                <div class="grid-content" >
                 <div style="display: flex;flex-direction: column;align-items: center;">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%;">烧伤</span>
                </div>
                <div style="display: flex;flex-direction: column;align-items: center;">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%;">撞伤</span>
                </div>
                </div>
            </el-col>
            <el-col :span="8">
                <div class="grid-content" >
                <div style="display: flex;flex-direction: column;align-items: center;">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%">骨折</span>
                </div> 
                <div style="display: flex;flex-direction: column;align-items: center;">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%">划伤</span>
                </div>
                </div>
            </el-col>
            <el-col :span="8">
                <div class="grid-content" >
                 <div style="display: flex;flex-direction: column;align-items: center;">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%">烧伤</span>
                </div>
                <div style="display: flex;flex-direction: column;align-items: center">
                <el-progress type="circle" :percentage="25" />
                <span style="margin-top: 5%;margin-bottom: 5%">枪弹伤</span>
                </div>
                </div>
            </el-col>
        </el-row>
    </div>
</template>

<script>    
export default {
    name:"Detail"
}
</script>

<style>
.el-row{
    align-items: center;
}
.el-col{
    height: 100% !important;
}
.grid-content{
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-evenly;
}
</style>