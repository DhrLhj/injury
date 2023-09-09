<template>
	<div class="record" style="width: 100%;height: 100%;">

		
		<input @click="startRecord" type="button" value="录音">
		<input @click="stopRecord" type="button" value="停止">


		<input @click="play" type="button" value="播放">
		<div class="record-play" v-show="isFinished">
			<h2>Current voice player is:</h2>
			<audio controls autoplay></audio>
		</div>
	</div>
</template>
 
<script>
import Record from "../common/record-sdk";
export default {
	name: "Record",
	data() {
		return {
			isFinished: false,
			audio: "",
			recorder: new Record()
		};
	},
	methods: {
		startRecord: function() {
			console.log("start to record now.");
			let self = this;
			self.isFinished = false;
			self.recorder.startRecord({
				success: res => {
					console.log("start record successfully123.");
				},
				error: res => {
					console.log("start record failed.");
				}
			});
		},
		stopRecord: function() {
			console.log("stop record now.");
			let self = this;
			self.isFinished = false;
			self.recorder.stopRecord({
				success: res => {
					//此处可以获取音频源文件(res)，用于上传等操作
					console.log("res---",res)
					console.log("res---",res.size)
					console.log("res---",res.type)

					console.log("++++++开始上传")
					var form = new FormData;
					form.append("file",res,"1.mp3"); 
					var xhr=new XMLHttpRequest();
					xhr.open("POST","http://localhost:9090/file/uploadImg");
					xhr.onreadystatechange=function(){
					if(xhr.readyState==4){
					alert(xhr.status==200?"上传成功":"测试请先打开浏览器控制台，然后重新录音，在network选项卡里面就能看到上传请求数据和格式了");
				}}
					xhr.send(form);
				},
				error: res => {
					console.log("stop record failed.");
				}
			});
			self.recorder.stop()
		},
		play: function() {
			console.log("play record now.");
			let self = this;
			self.isFinished = true;
			self.audio = document.querySelector("audio");
			self.recorder.play(self.audio);
		}
	}
};
</script>