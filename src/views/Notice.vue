<template>
  <div style="height: 95%;width: 70%;margin:1% 15% 1% 15%;border: 2px solid #dcdfe6;
  color: #306eff;
  border-radius: 1%;
  ">
  <div style="margin-left: 1%;margin-top: 2%;">
    <span style="font-size: 2vh;margin-left: 2%">共2条内容</span>
    <div style="margin-top: 1%;">
      <span style="font-size: 3.5vh;">【系统】Hi，新士兵！欢迎体验紧急救援教学系统</span>
      <el-scrollbar height="20vh" style="background-color: #dcdfe6;margin-left: 1.2%;margin-right: 10%;border-radius: 10px;">
          <el-text size="default" tag="p" style="font-size: 3vh;margin-left: 1%;">
              Hi,新士兵！欢迎体验紧急救援教学系统。本系统是由北京科技大学与xx战区联合制作<br>
            并发布的，该系统还处于测试阶段......
          </el-text>
      </el-scrollbar>
    </div>
    <div style="margin-top: 1%;">
      <span style="font-size: 3.5vh;">【系统】士兵紧急救援系统使用教程</span>
      <el-scrollbar height="20vh" style="background-color: #dcdfe6;margin-left: 1.2%;margin-right: 10%;border-radius: 10px;">
          <el-text size="default" tag="p" style="font-size: 3vh;margin-left: 1%;">
              Hi,新士兵！欢迎体验紧急救援教学系统。本条消息详细地介绍了如何使用本系统，快<br>
              来看看吧！
          </el-text>
      </el-scrollbar>
    </div>
  </div>
  <div style="height: 25%;display: flex;
    justify-content: center;
    align-items: flex-end;">
    <el-pagination
      background
      v-model:current-page="currentPage1"
      :page-size="100"
      :page-count="1"
      :small="small"
      :disabled="disabled"
      prev-text="上一页"
      next-text="下一页"
      layout="total, prev, pager, next"
      :total="2"
      @size-change="handleSizeChange"
      @current-change="handleCurrentChange">
      </el-pagination>

</div>
  </div>
</template>

<script>
import { ref } from 'vue'
export default {
    name:"Notice",
    data(){
      return{
        small :ref(false),
      }
    }

}
</script>

<style>
p{
    line-height: 6vh;
}
.el-text{
  color: #306eff;
}
</style>