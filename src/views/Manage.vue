<template>
      <el-container style="height: 100%;">


        <el-header style="height: 15%;padding: 0px;background-color: #036eff;">
            <Header />
        </el-header>
        <el-main  style="height: 85%;padding: 0;">
            <!-- <Home /> -->
            <router-view />
        </el-main>
      </el-container>
</template>

<script>
import Header from '@/components/Header.vue';
export default {
  data(){
    return{
    }
  },
    components: {
    Header,
  },
  mounted(){
    // this.redir()
  },
  methods:{
    redir(){
      // console.log("我要跳转了")
      // router.push(this.path)
    }
  }
}
</script>

<style>

</style>