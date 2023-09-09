<template>
    
    <el-container>
      <el-header>
        <Header/>
      </el-header>
      <el-container>
        <el-aside width="20%">
            <div style="display: inline-block;height: 96%;width: 15%;margin-left: 2%;margin-top: 2%;">
        <div style="border: 2px solid var(--el-border-color);height: 96%;border-radius: 2vh;">
        <div style="display: flex;align-items: flex-end;margin-top: 6%;margin-left: 6%;">
        <el-progress type="circle" :percentage="25">
            <template #default="{ percentage }">
              <div class="circleText">
                <span class="percentage-label">已学</span>
              </div>
              <div class="circleText" style="margin-top: 20%">
                <span class="percentage-value">{{ percentage }}%</span>
              </div>
            </template>
          </el-progress>
          <router-link to="/detail" id="detail">
          <span  style="color: #306eff;margin-left: 3%;">查看详情</span>
        </router-link>
        </div>
        <div style="position: relative;font-size: 2vh;height: 5%;display: flex;align-items: center;margin-top: 10%;background-color: rgb(229, 240, 255);;">
            <span style="margin-left: 5%;">目录</span>
            <el-select v-model="value" clearable placeholder="请选择" style="width: 50%;position: absolute;right: 0;">
                    <el-option
                    v-for="item in options"
                    :key="item.value"
                    :label="item.label"
                    :value="item.value"
                    />
                </el-select>
        </div>
        
        <div class="demo-collapse" style="margin-left: 2%;">
            <el-collapse v-model="activeNames" accordion>
            <el-menu
                active-text-color="#036eff"
                class="el-menu-vertical-demo"
                :default-active="1"
                text-color="black"
                @open="handleOpen"
                >
            <el-collapse-item title="烧伤" name="1">
                <div  class="left">
                <!-- <router-link to="/study/first"> -->
                <el-menu-item @click="moveByid(index)" id="leftItem" value="1" index="1" ><el-icon><VideoPlay /></el-icon>如何辨别烧伤？</el-menu-item>
                <!-- </router-link> -->
                <el-menu-item @click="moveByid(index)" id="leftItem1" index="3"><el-icon><PictureFilled /></el-icon>救治方案</el-menu-item>
            </div>
            </el-collapse-item>
            
            <el-collapse-item title="骨折" name="2">
                <div  class="left">
                <el-menu-item id="leftItem" index="3"><el-icon><VideoPlay /></el-icon>如何辨别骨折？</el-menu-item>
                <el-menu-item id="leftItem" index="4"><el-icon><PictureFilled /></el-icon>救治方案</el-menu-item>
                </div>
            </el-collapse-item>
        

            <el-collapse-item title="撞伤" name="3">
                <div  class="left">
                <el-menu-item id="leftItem" index="5"><el-icon><VideoPlay /></el-icon>如何辨别撞伤？</el-menu-item>
                <el-menu-item id="leftItem" index="6"><el-icon><PictureFilled /></el-icon>救治方案</el-menu-item>
                </div>
            </el-collapse-item>
        
            <el-collapse-item title="划伤" name="4">
                <div  class="left">
                <el-menu-item id="leftItem" index="7"><el-icon><VideoPlay /></el-icon>如何辨别划伤？</el-menu-item>
                <el-menu-item id="leftItem" index="8"><el-icon><PictureFilled /></el-icon>救治方案</el-menu-item>
                </div>
            </el-collapse-item>
            <el-collapse-item title="枪弹伤" name="5">
                <div  class="left">
                <el-menu-item id="leftItem" index="9"><el-icon><VideoPlay /></el-icon>如何辨别枪弹伤？</el-menu-item>
                <el-menu-item id="leftItem" index="10"><el-icon><PictureFilled /></el-icon>救治方案</el-menu-item>
                </div>
            </el-collapse-item>
        </el-menu>
            </el-collapse>
        </div>


    </div>

    </div>
        </el-aside>
        <el-main>Main</el-main>
      </el-container>
    </el-container>
   

</template>

<script>
// import Vedio from '@/components/Vedio.vue';
// import Studymovie1 from '@/components/Studymovie1.vue'
import Header from '@/components/Header.vue';
import { ref } from 'vue';
export default {
  name: "Left",
  data(){
    return{
        index:'1',
        activeNames:"",
        oneClick:false,
        videourl:"",
        rightShow:1,
        selOne:"",
        twoClick:false,
        value:ref("全部"),
        num:[1],
        options:[{value: '全部',label: '全部'},{value: '烧伤',label: '烧伤'},{value: '炸伤',label: '炸伤'},]
    }
  },
    components:{
        Header,
        // Vedio,
        // Test,
        // Studymovie1,
    },
  methods:{
    changeVideo(newUrl){
            console.log(newUrl,"----------")
            this.videourl=newUrl;
            // this.$refs.videoPlayer.load();
        }
    ,
    selected(index){
        console.log(index)
        
    },
    handleOpen(key, keyPath) {
    console.log("------")
    console.log(key)
},  
moveByid(index){
    console.log("++++++"+index)
    // this.index='2';
  }
}}
</script>

<style>
.el-progress__text {
  display: flex;
  flex-direction: column;
}
.circleText {
  color: #036eff;
}
.example-showcase .el-dropdown-link {
  cursor: pointer;
  color: var(--el-color-primary);
  display: flex;
  align-items: center;
}
.el-dropdown-link:hover{
    border: none;
    margin: 0;
}
.el-collapse-item__content {
    padding-bottom: 0;
}
.collapsetext>span{
    cursor: pointer;
    margin-left: 15%;
    display: block;
    font-size: 1.6vh;
}
.movie,.photo{
    color: #036eff;
}
#leftItem.is-active {
    background-color: #ffffff !important;
}
#leftItem{
    background-color: #ffffff !important;
    font-size: 1.6vh;
    height: 5vh;
}
.ulStyle{
    list-style: none;
    cursor: pointer;
}
.vedio{
    width: 90%;
    height: 71vh;
    margin-bottom: 4vh;
}
</style>