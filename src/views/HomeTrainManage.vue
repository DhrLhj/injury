<template>
    <el-container style="height: 100%;">
        <el-container>
        <el-aside width="20%">
          <Left/>
        </el-aside>
        <el-main>
            <!-- <router-link/> -->
        </el-main>
      </el-container>
    </el-container>
</template>

<script>
import Header from '@/components/Header.vue';
export default {
    name:"StudyManage",
  components: {
        Header,
        // Home,
},

}
</script>

<style>

</style>