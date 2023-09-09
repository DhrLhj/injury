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
import Study from '@/views/Study.vue';
import Left from '@/views/Left.vue';

// import Home from '@/views/Home.vue';
export default {
    name:"StudyManage",
  components: {
  Header,
  Study,
  Left,
  // Home,
},

}
</script>

<style>

</style>