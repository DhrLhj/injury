<template>
  <div style="height: 100%;">
    <div style="height: 51%;display:flex;">
      <div style="height: 110%;display:flex;padding-left:1.1%;align-items: center;">
        <img src="../assets/icon/特种医学与军事医学科.png" alt="" style="height:80%;">
      </div>
      <div style="height: 100%;display: flex;padding-top: 0.5%;padding-left:0.5%;align-items: center;">
        <p style="font-size:4.7vh;color: white;">&#8197;&#8197;士&#8197;兵&#8197;紧&#8197;急&#8197;救&#8197;援&#8197;教&#8197;学&#8197;系&#8197;统
        <!-- <el-divider style="display: inline;" direction="vertical"/> -->
        </p>
      </div>
    </div>
    <div style="height: 50%;">
      <div style="height: 100%;align-items: center;">
        <el-menu
          class="el-menu-demo"
          text-color="#ffffff"
          active-text-color="#ffffff"
          mode="horizontal"
          :default-active="activeIndex"
          style="height: 100%;"
          @select="handleSelect"
        >
        <router-link to="/home" style="margin-left: 6.3%;">
        <el-menu-item to="/home" index='1'>首&#8198;页</el-menu-item>
        </router-link>
        <!-- <router-link to="/change" style="margin-left: 6.3%;">
        <el-menu-item to="/change" index='1'>缩放</el-menu-item>
        </router-link> -->
        
        <!-- <router-link to="/body" style="margin-left: 6.3%;">
        <el-menu-item to="/body" index='1'>识别验证</el-menu-item>
        </router-link> -->
        
        <router-link to="/notice" style="margin-left: 4.8%;">
        <el-menu-item ref="mybtton" id="system" index="2" >系&#8198;统&#8198;通&#8198;知</el-menu-item>
        </router-link>
        <el-input id="startTime" placeholder="战场上如何快速止血" style="width: 45%;margin-top: 0.2%;margin-left: 5.5%;height: 75%;">
          <template #suffix>
            <!-- <div class="suffix" style="height: 100%;"><img src="../assets/icon/语音.png" alt=""></div> -->
          </template>
        </el-input>
        <router-link to="/information" style="margin-left: 4%;">
        <el-menu-item index="3" style="margin-left: 4%;">个&#8198;人&#8198;中&#8198;心</el-menu-item>
        </router-link>
        <el-menu-item style="margin-right:0.5%;width: 6%;margin-left:2%">
          <el-button id="leftButton" style="width: 100%;background-color: #036eff;color:#ffffff;font-size: 100%;height: 100%;">登录</el-button>
        </el-menu-item>
        <el-menu-item style="margin-right:0.5%;width: 6%;">
          <el-button style="width: 100%;background-color: #ffffff;font-size: 100%;height: 100%;">注册</el-button>
        </el-menu-item>
        <img src="../assets/icon/士兵.png" alt="" style="height:95%;border: 2px solid #ffffff;border-radius: 100%;align-content: right;">
      </el-menu>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
export default {
    name:"Header",
    data(){
      return{
        activeIndex : ref('1')
      }
    },
    methods:{
      changeborder(){
        console.log('改变了路由')
        this.index="/changeborder"
      },
      handlenext() {
            document.querySelector("#TrainingResults").click();
        },
    },
    created() 
    {


        

    },
    mounted() {
    }
}
</script>

<style>
.el-menu-item{
  padding: 0px;
  font-size: 3vh;
  background-color: #036eff !important;
  height: 100%;
}
.el-menu{
  background-color: #036eff;
}
.el-menu-item.is-active{
  background-color: #036eff !important;

}
.suffix{
  float: right;
  position: absolute;
  left: 0;
}
.el-menu-item>.el-button{
  /* border: 1px solid #ffffff; */
  border-radius: 33%;
  color:#036eff;
  padding: 0px;
}
#leftButton{
  border: none;
}
.router-link-active{
  text-decoration: none;
  border-bottom: 3px solid #ffffff;
  padding: -3px;
}
a{
  text-decoration: none;
  height: 100%;
  /* color: black; */
}
.el-menu--horizontal{
  align-items: center;
}
</style>