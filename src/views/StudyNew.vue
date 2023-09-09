<template>
    
    <el-container  style="height: 100%;">
        <el-header style="height: 15%;padding: 0px;background-color: #036eff;">
            <Header />
        </el-header>
      <el-container style="width: 100%;height: 100%;">
      <el-aside style="width:18%;height: 100%;margin: 1% 0 0.5% 1%;">

        <div style="display: inline-block;height: 100%;width: 100%;">
        <div style="border: 2px solid var(--el-border-color);height: 96%;border-radius: 2vh;">
        <div style="display: flex;align-items: flex-end;margin-top: 6%;margin-left: 6%;">
            <div style="height: 15vh;width: 94%;background-color: rgba(210, 220, 231);border-radius: 10px;border: 1px solid black;display: flex;align-items: center;
              justify-content: center;">
              <b style="color: #5c7aa5;font-size: 2vw;">知识库</b>
            </div>
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

                :default-active=activeNames
                active-text-color="#036eff"
                class="el-menu-vertical-demo"
                text-color="black"
                @open="handleOpen"
                >
            <el-collapse-item title="烧伤" name="1">
                <div class="left" @click="shaoshang">
                <router-link to="/study/right/1">
                <el-menu-item id="leftItem1"  index="1" ><el-icon><PictureFilled /></el-icon>等级1-烧伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/burn">
                <el-menu-item id="leftItem2" index="2"><el-icon><VideoPlay /></el-icon>等级2-烧伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/burnDrug">
                <el-menu-item id="leftItem2" index="2-1"><el-icon><Menu /></el-icon>等级3-烧伤救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>
            <el-collapse-item title="骨折" name="2">
                <div class="left" id="guzhe12" @click="guzhe">
                  <router-link to="/study/right/2">
                <el-menu-item id="leftItem3" index="3"><el-icon><PictureFilled /></el-icon>等级1-骨折救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/fracture">
                <el-menu-item id="leftItem4" index="4"><el-icon><VideoPlay /></el-icon>等级2-骨折救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/fractureDrug">
                <el-menu-item id="leftItem2" index="2-1"><el-icon><Menu /></el-icon>等级3-骨折救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>
        

            <el-collapse-item title="撞伤" name="3">
                <div  class="left"  @click="zhuangshang">
                <router-link to="/study/right/3">
                <el-menu-item id="leftItem5" index="5"><el-icon><PictureFilled/></el-icon>等级1-撞伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/bruise">
                <el-menu-item id="leftItem6" index="6"><el-icon><VideoPlay/></el-icon>等级2-撞伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/bruiseDrug">
                <el-menu-item id="leftItem2" index="2-1"><el-icon><Menu /></el-icon>等级3-撞伤救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>
        
            <el-collapse-item title="划伤" name="4">
                <div class="left" @click="huashang">

                <router-link to="/study/right/4">
                <el-menu-item id="leftItem7" index="7"><el-icon><PictureFilled/></el-icon>等级1-划伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/scratches">
                <el-menu-item id="leftItem8" index="8"><el-icon><VideoPlay /></el-icon>等级2-划伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/scratchesDrug">
                <el-menu-item id="leftItem2" index="2-1"><el-icon><Menu /></el-icon>等级3-划伤救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>
            <el-collapse-item title="枪弹伤" name="5">
                <div  class="left"  @click="qiangdanshang">

                <router-link to="/study/right/5">
                <el-menu-item id="leftItem9" index="9"><el-icon><PictureFilled /></el-icon>等级1-枪弹伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/gunshot">
                <el-menu-item id="leftItem10" index="10"><el-icon><VideoPlay/></el-icon>等级2-枪弹伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/gunshotDrug">
                <el-menu-item id="leftItem10" index="10-1"><el-icon><Menu /></el-icon>等级3-枪弹伤救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>

            <el-collapse-item title="炸伤" name="6">
                <div  class="left"  @click="zhashang">

                <router-link to="/study/right/6">
                <el-menu-item id="leftItem11" index="9"><el-icon><PictureFilled /></el-icon>等级1-炸伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/explosion">
                <el-menu-item id="leftItem12" index="10"><el-icon><VideoPlay/></el-icon>等级2-炸伤救治方案</el-menu-item>
                </router-link>
                <router-link to="/study/right/explosionDrug">
                <el-menu-item id="leftItem10" index="10-1"><el-icon><Menu /></el-icon>等级3-炸伤救治方案</el-menu-item>
                </router-link>
              </div>
            </el-collapse-item>
        </el-menu>
            </el-collapse>
        </div>


    </div>
        </div>

      </el-aside>
      <el-main style="width: 70%;height: 100%;padding-top: 0;">
        <router-view />
      </el-main>
    </el-container>
    </el-container>





</template>

<script>
import Header from '@/components/Header.vue'
import { ref } from 'vue';
export default {
  name: "Study",
  data(){
    return{
        // activeNames:"1",
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
    },
  methods:{
    selected(index){
        console.log(index)
    },
    handleOpen(key, keyPath) {
    console.log("xxx被打开了------")
    console.log(key)
  },
  moveByid(index){
    console.log("++++++"+index)
  },
  onChange(){
    this.activeNames='';
  },
  shaoshang(){
    this.activeNames='1'
  },
  guzhe(){
    this.activeNames='2'
  },
  zhuangshang(){
    this.activeNames='3'
  },
  huashang(){
    this.activeNames='4'
  },
  qiangdanshang(){
    this.activeNames='5'
  },
  zhashang(){
    this.activeNames='6'
  },
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
#leftItem1.is-active {
    background-color: #ffffff !important;
}
#leftItem1,#leftItem2,#leftItem3,#leftItem4,#leftItem5,#leftItem6,#leftItem7,#leftItem8
,#leftItem9,#leftItem10,#leftItem11,#leftItem12{
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
.router-link-active{
  text-decoration: none;
}
a{
  text-decoration: none;

}
</style>