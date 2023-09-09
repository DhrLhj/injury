<template>
      <video ref="videoPlayer"  controls :src="videourl"></video>
</template>

<script>
export default {
    name:"Studymovie1",
    data(){
        return{
            videourl:"http://clips.vorwaerts-gmbh.de/big_buck_bunny.mp4"
        }
    },
    methods:{
        changeVideo(newUrl){
            console.log(newUrl,"----------")
            this.videourl=newUrl;
            this.$refs.videoPlayer.load();
        }
    }
}
</script>

<style>

</style>