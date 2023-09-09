<template>
  <div style="width: 100%;height: 100%;">
                <div style="margin:10vh 0 0 10vh;width:70% ;">
                <span>个人信息</span>
                <span style="float: right;color: #306eff;"><el-icon><Edit/></el-icon>编辑</span>
                <div style="height: 50vh;w;margin-top: 2vh;background-color: rgb(238, 239, 240);border-radius: 20px;">
                    
                    <el-row :gutter="20">
                        <el-col :span="2"><div class="grid-content ep-bg-purple" /></el-col>
                        <el-col :span="5"><div class="grid-content ep-bg-purple" />
                            <div style="margin-top: 40%;">
                                姓名：许三多
                            </div>
                            <div style="margin-top: 30%;">
                                所属战区：东部战区
                            </div>
                            <div style="margin-top: 30%;">
                                服役年限：3年
                            </div>
                        </el-col>
                        <el-col :span="8"><div class="grid-content ep-bg-purple" /></el-col>
                        <el-col :span="9"><div class="grid-content ep-bg-purple" />
                            <div style="margin-top: 20%;">
                                性别：男
                            </div>
                            <div style="margin-top: 15%;">
                                学历：无
                            </div>
                        </el-col>
                    </el-row>

                </div>
                </div>
            </div>
</template>

<script>



export default {
    name:"Information",
    data(){
        return{
        }
    },
    methods: {
        handlenext() {
            document.querySelector("#TrainingResults").click();
        },
        
    },
    created() 
    {
        // setTimeout(() => {
        this.$ws.addEventListener('message', (event) => {
        // 处理 WebSocket 消息
        const message = event.data;
        console.log('WebSocket消息：', message);
        
        if (message==='1') {
                this.handlenext(); 
            }
        })
        // }, 3000);  
        

    },
    mounted() {
    }

}
</script>

<style>

</style>