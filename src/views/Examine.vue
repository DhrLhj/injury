<template>
    <div style="width: 40%;height: 70%;margin-left:30%;margin-top: 5%;border: 2px solid #dcdfe6;border-radius: 10px;">
        <div style="width: 100%;height: 100%;">
            <div style="display: flex;height: 30%;width: 100%;font-size: 3vh;
                justify-content: center;
                align-items: center;">
                <span>考核测试</span>
            </div>

        <div style="height: 40%;margin:0 10% 0 10%;width: 80%;background-color: rgb(229, 240, 255);display: flex;
                    flex-direction: column;">
            <div style="height: 30%;width: 100%;display: flex;
                    justify-content: center;">
                <span style="margin-top: 2%;height: 10%;">————考核须知————</span>
            </div>
            <div style="height: 50%;width: 100%;margin-left: 3%;display: flex;flex-direction:column">
                <span style="margin-top: 1%;">考核时间：120分钟</span>
                <span style="margin-top: 1%;">考核提醒：判断、选择</span>
                <span style="margin-top: 1%;">提交次数：作答完成后只有一次提交机会</span>
            </div>
        </div>
        <router-link to="/ExamineInner" id="ExamineInner">
        <el-button type="primary" style="margin-left: 43%;margin-top: 5%;">开始考核</el-button>
        </router-link>
        </div>
    </div>
</template>

<script>
export default {
    name:"Examine",
    data(){
        return{
        }
    },
    methods: {
        hh() {
            router.push("/ExamineInner");
        
    },
    },
    created() 
    {
        this.$ws.addEventListener('message', (event) => {
        // 处理 WebSocket 消息
        const message = event.data;
        console.log('WebSocket消息：', message);
        
        if (message === 'hh') {
            this.hh();
        }
        })
        

    },
    mounted() {
    }

}
</script>

<style>

</style>