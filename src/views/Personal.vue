<template>
    <el-container  style="height: 100%;">
        <el-header style="height: 15%;padding: 0px;background-color: #036eff;">
            <Header />
        </el-header>
      <el-container style="width: 100%;height: 100%;">
        <el-aside width="20%">
            <div style="width:92%;height: 94%;margin: 4%;border-radius: 10px;border: 2px solid #dcdfe6;">
                <div style="height: 25vh;display: flex;justify-content: center;align-items: center;
                flex-direction: column;">
                    <img style="width: 10vh;height: 10vh;" src="../assets/icon/士兵.png" alt="">
                    <span >士兵许三多</span>
                </div>
                <div class="person">
                    <el-menu
                    style="background-color: #ffffff;"
                    :default-active="activeIndex"
                    class="el-menu-demo"
                    mode="vertical"
                    text-color="balck">
                    <router-link to="/information">
                    <el-menu-item @click="personaldata" index="1"><img id="personaldata" style="width: 3vh;margin-right: 0.5vw;line-height: 5vh;" src="../assets/icon/个人资料.png"/> 个人资料</el-menu-item>
                    </router-link>
                    <!-- <router-link to="/trainingResults">
                    <el-menu-item @click="TrainingResults" index="2"><img id="TrainingResults" style="width: 3vh;margin-right: 0.5vw;line-height: 5vh;" src="../assets/icon/成绩.png"/> 训练成绩</el-menu-item>
                    </router-link>
                    <router-link to="/studySchedule">
                    <el-menu-item @click="studySchedule" index="3"><img id="studySchedule" style="width: 3vh;margin-right: 0.5vw;line-height: 5vh;" src="../assets/icon/进度.png"/> 学习进度</el-menu-item>
                    </router-link> -->
                </el-menu>
                </div>
            </div>
      </el-aside>
      <el-main style="width: 70%;padding-top: 0;">
        <router-view style="width: 100%;"/>
      </el-main>
    </el-container>
    </el-container>




</template>

<script>
import Header from '@/components/Header.vue';
import { ref } from 'vue'
export default {
    name:"Personal",
    data(){
        return{
            activeIndex : ref('1')
        }
    },
    components:{
        Header,
    },
    methods:{
        personaldata(){
            this.activeIndex='1'
        },
        TrainingResults(){
            this.activeIndex='2'
        },
        studySchedule(){
            this.activeIndex='3'
        }
    }
}
</script>

<style scoped>
.el-menu-demo{
    background-color: #ffffff;
}
.el-menu-item{
    background-color: #ffffff !important;
    font-size: 2vh !important;
}
.el-menu-item.is-active{
    background-color: aliceblue !important;
    border-left: 4px solid #306eff;
}
.router-link-active{
  text-decoration: none;
}
a{
  text-decoration: none;
  color: black;
}
.el-menu--horizontal>.el-menu-item.is-active {
    border-bottom: 2px solid var(--el-menu-active-color);
}
</style>