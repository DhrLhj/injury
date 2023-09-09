<template>
    <div style="height: 100%;margin-top:0.5% ;">
    <div style="color: #036eff;font-size: 2.5vh;">
    <span style="margin-left: 2%;">训练模式</span>
    <span style="float: right;margin-right: 2%;">剩余时间： 120分钟</span>
    <el-divider style="margin: 0;"/>
    </div>
    <div style="display: inline-block;height: 100%;margin-left: 2%;margin-top: 0.8%;">
    <span style="color: #036eff;">0/10题</span>
    <ul class="ulStyle">
        <!-- <li v-for="(item,index) in options" :class='{isActive:index==isactive}' @click="selected(index)" :key="item">{{ item }}</li> -->
        <li v-for="(item,index) in options" :id="'index'+(index+1)" :class='{isActive:index==isactive}' @click="selected(index)" :key="item">{{ item }}</li>
    </ul>
    </div>
    <div style="display: inline-block;float: right;border-left: 2px solid #dcdfe6;width: 92%;height: 100%;">
        <div id="rightSel" style="margin-left: 10%;margin-top: 1%;">
            <span style="font-size: 3vh;">判断题</span>
            <div style="display: flex;font-size: 1.8vh;align-items: center;">
            <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
            <span style="margin-left: 0.5%;">辨别下图伤情</span>
            </div>

            <div>
            <el-image class="imgTwo" :src="require('@/assets/imagebox/burn/1.jpg')" alt="" style="height: 60vh;width: 90%;" />
            </div>

            
            <span id="determine" :class='{acTived:show==false}' :v-model="msg" v-on:click="sendVoice()" style="display: block;font-size: 4vh;cursor: pointer;color: #036eff;border-radius: 10px;margin-top: 2vh;width: 90%;height: 8vh;border: 2px solid #dcdfe6;">
            <img v-show="show" style="width: 5vh;position: relative;top:1vh;left: 1vh;" src="../assets/icon/语音.png" alt=""/>
            <img v-show="!show" style="width: 5vh;position: relative;top:1vh;left: 1vh;" src="../assets/icon/播放中.png" alt=""/>
            <template v-if="show">
                <span class="msg">{{ msg }}</span>
            </template>
            <template v-else>
                <span class="msg">{{ afterMsg }}</span>
            </template>
            </span>
            <div :class="{isShow:showValue==true}" style="margin-top: 2vh;width: 90%;display: flex;align-items: center;">
                <div style="display: inline-block;">
                    <el-image :class="{isshow:isTrue==true}" style="height: 5vh;" :src="require('@/assets/icon/正确.png')" alt=""/>
                    <el-image :class="{isshow:isfalse==true}" style="height: 5vh;" :src="require('@/assets/icon/错误.png')" alt=""/>
                </div>
                <div v-show="value" style="display: inline-block;margin-left:1%;height: 5vh;font-size: 3vh;color: #036eff;">{{ answer }}</div>
            </div>


            <!-- <el-button @click="playVedio()" type="primary">点击查看操作视频</el-button> -->

            <div style="display: flex;margin-top: 1%;font-size: 1.8vh;align-items: center;">
            <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
            <span style="margin-left: 0.5%;">请选择正确的救治方案</span>
            </div>
            <div style="width: 90%;">
                <el-row style="">
                    <el-col style="margin-right: 2%;margin-top: 2%;" v-for="(image,index) in imagebox" :key="index" :span="8" :index="index" @click="selecte(index)">
                        <el-card>
                            <el-image class="imgOne" :src="image" alt="" style="display: flex;height: 25vh;" />
                        </el-card>
                    </el-col>
                </el-row>
            </div>
            <div style="display: flex;">

                
            <!-- <div style="width: 30%;">
            <span :class='{acTived:show==false}' :v-model="msg" v-on:click="startRecord()" style="display: block;font-size: 3vh;cursor: pointer;color: #036eff;border-radius: 10px;margin-top: 2vh;width: 100%;height: 8vh;border: 2px solid #dcdfe6;">
            <img v-show="show" style="width: 5vh;position: relative;top:1vh;" src="../assets/icon/语音.png" alt=""/>
            <img v-show="!show" style="width: 5vh;position: relative;top:1vh;left: 1vh;" src="../assets/icon/播放中.png" alt=""/> -->
            <!-- {{ msg }} -->
            <!-- <template v-if="show">
                <span class="msg">{{ voiceMsg }}</span>
            </template>
            <template v-else>
                <span class="msg">{{ afterMsg }}</span>
            </template>
            </span>
            <span>
                <el-button type="primary" @click="stopRecord()">停止录音</el-button>
            </span>
            </div> -->
            
            <div  style="width: 30%;">
            <span  id="select" :class='{acTivedTwo:showTwo==false}' :v-model="msg" v-on:click="sendVoiceTwo()" style="display: block;font-size: 3vh;cursor: pointer;color: #036eff;border-radius: 10px;margin-top: 2vh;width: 100%;height: 8vh;border: 2px solid #dcdfe6;">
            <img v-show="showTwo" style="width: 5vh;position: relative;top:1vh;" src="../assets/icon/语音.png" alt=""/>
            <img v-show="!showTwo" style="width: 5vh;position: relative;top:1vh;left: 1vh;" src="../assets/icon/播放中.png" alt=""/>
            <!-- {{ msg }} -->
            <template v-if="showTwo">
                <span class="msg">{{ voiceMsgTwo }}</span>
            </template>
            <template v-else>
                <span class="msg">{{ afterMsgTwo }}</span>
            </template>
            </span>
            <!-- <span>
                <el-button type="primary" @click="stopRecord()">停止录音</el-button>
            </span> -->
            </div>


            <div style="margin-left: 20%;display:grid;width: 30%;">
            <span :class='{showIcon:show==false}' :v-model="msg" v-on:click="dialogVisible = true,initCamera()" style="display: flex;align-items: center;font-size: 3vh;cursor: pointer;color: #036eff;border-radius: 10px;margin-top: 2vh;width: 100%;height: 8vh;border: 2px solid #dcdfe6;">
            <el-icon size="40"><CameraFilled /></el-icon>
            <span class="msg">{{ photoMsg }}</span>
            </span>
            </div>
        </div>
            <div :class="{isShow:showValueTwo==true}" style="margin-top: 2vh;width: 90%;display: flex;align-items: center;">
                <div style="display: inline-block;">
                    <el-image :class="{isshow:isTrueTwo==true}" style="height: 5vh;" :src="require('@/assets/icon/正确.png')" alt=""/>
                    <el-image :class="{isshow:isfalseTwo==true}" style="height: 5vh;" :src="require('@/assets/icon/错误.png')" alt=""/>
                </div>
                <div v-show="valueTwo" style="display: inline-block;margin-left:1%;height: 5vh;font-size: 3vh;color: #036eff;">{{ answer }}</div>
            </div>


            <div>
            <div :class='{isShowVedio:showvedio==true}'>
                <span style="font-size: 2.5vh;">如何辨别烧伤？</span>
                <div style="display: flex;margin-top: 1%;font-size: 1.8vh;align-items: center;">
                <el-divider style="margin: 0;border-width: 3px;border-color: black;" direction="vertical" />
                <span style="margin-left: 0.5%;">请选择正确的救治方案</span>
                </div>
                
                <video ref="videoPlayer" controls style="width: 90%;height: 88%;">
                <source :src="videourl" type="video/mp4">
                </video>

            </div>
            </div>
        </div>
    
    </div>


    <el-dialog
    v-model="dialogVisible"
    title="Tips：请允许访问摄像头权限进行拍照"
    width="40%"
  > 
    <div style="width: 90%;margin-left: 10%;">

    <video id="videoCamera" style="width: 100%;padding: 3%;border-radius: 5%;border: 1px solid #036eff;" ref="videoElement" autoplay></video>
    


      <el-button type="primary" @click="takePhoto()" style="width: 100%;height: 20%;">拍照</el-button>
      <canvas style="width: 100%; height: 40vh;"
      ref="canvasElement"></canvas>
      <!-- <img :src="photoUrl" v-if="photoUrl" alt="拍摄照片"> -->
    </div>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="dialogVisible = false,sendPhoto()">
          提交
        </el-button>
      </span>
    </template>
  </el-dialog>



    </div>
</template>


<script>
import Record from "../common/record-sdk";
import { ref } from 'vue'
import {sendPhoto} from '@/api/api'
// import Vedio from '@/components/Vedio.vue';
export default {
    name:"ExamineInner",
    data(){
        return {
            isactive:'0',
            show:true,
            showTwo:true,
            showValue:true,
            showValueTwo:true,
            isTrue:true,
            isfalse:true,
            isTrueTwo:true,
            isfalseTwo:true,
            value:false,
            valueTwo:false,
            showvedio:true,
            selectedNo:["1","2","3","4"],
            options:["1","2","3","4","5","6","7","8","9","10"],
            voiceMsg:"请点击使用语音进行回答",
            voiceMsgTwo:"请点击使用语音进行回答",
            photoMsg:"请点击拍摄手势回答",
            beforeMsg1:"请选择正确的救治方案或语音输入 🔊",
            afterMsg:"  这个是烧伤",
            afterMsgTwo:"  这道题选图1",
            answer:"回答正确",
            dialogVisible:ref(false),
            photoUrl: null,
            thisVideo:null,
            isFinished: false,
			audio: "",
			recorder: new Record(),
            videourl:"/videos/划伤救治.mp4",
            imagebox:[
            require('@/assets/imagebox/burn/1.jpg'),
            require('@/assets/imagebox/bruise/2.jpg'),
            require('@/assets/imagebox/burn/3.jpg'),
            require('@/assets/imagebox/gunshot/2.jpg'),
            ],

        }
    },
    components:{
        // Vedio,
    },

    methods:{
        selected(index){
            console.log(index);
            this.isactive = index;
            const target = document.getElementById('rightSel');
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
            this.show=true
            this.showValue=!this.showValue
            this.isTrue=true
            this.value=false
            // -------------
            this.showTwo=true
            this.showValueTwo=!this.showValueTwo
            this.isTrueTwo=true
            this.valueTwo=false
        },
        sendVoice(){
            console.log("发送了一段语音")
            this.show = !this.show
            console.log("+++"+this.showValue)
            this.showValue=!this.showValue
            this.isTrue=false
            this.value=true
        },
        sendVoiceTwo(){
            console.log("发送了一段语音")
            this.showTwo = !this.showTwo
            console.log("+++"+this.showValueTwo)
            this.showValueTwo=!this.showValueTwo
            this.isTrueTwo=false
            this.valueTwo=true
        },
        sendPhoto(){
            const canvasElement = this.$refs.canvasElement;

            // 获取Canvas中的图像数据
            const imageData = canvasElement.toDataURL('image/png');

            // // 显示拍摄的照片
            // this.photoUrl = imageData;

            const file = imageData
            const time = (new Date()).valueOf()
            // const name = time + '.png'
            const name = time
            console.log("name"+name)
            console.log("file+++"+file)
            const conversions = this.base64ToFile(file, name)
            const data = new FormData()

            console.log()
            data.append('file', conversions)
            
            sendPhoto(data).then(res=>{
                if(res.code==='200'){
                    console.log("成功了")
                    this.$message.success("提交成功")
                    this.value=true
                    this.isTrue=false
                    
                }else{
                    this.$message.success("图片未能识别,请重新拍照提交")
                    console.log("图片未能识别")
                }
                this.stopNavigator()
            })
        },
        selecte(index){
            console.log("选择了第"+index+"个")
        },
        playVedio(){
            this.showvedio=false
            this.$message.success("请翻到底部查看操作视频")
        },
      initCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
          .then(stream => {
            const videoElement = this.$refs.videoElement;
            videoElement.srcObject = stream;
            videoElement.play();
          })
          .catch(error => {
            console.error('无法访问摄像头：', error);
            this.$message.error("无法访问摄像头：")
          });
      },
        takePhoto() {
        const videoElement = this.$refs.videoElement;
        const canvasElement = this.$refs.canvasElement;
        const context = canvasElement.getContext('2d');

        // 将视频流的画面绘制到Canvas中
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        },
        // base64图片转file的方法（base64图片, 设置生成file的文件名）
        base64ToFile(base64, fileName) {
        // 将base64按照 , 进行分割 将前缀  与后续内容分隔开
        const data = base64.split(',')
        // 利用正则表达式 从前缀中获取图片的类型信息（image/png、image/jpeg、image/webp等）
        const type = data[0].match(/:(.*?);/)[1]
        // 从图片的类型信息中 获取具体的文件格式后缀（png、jpeg、webp）
        const suffix = type.split('/')[1]
        // 使用atob()对base64数据进行解码  结果是一个文件数据流 以字符串的格式输出
        const bstr = window.atob(data[1])
        // 获取解码结果字符串的长度
        let n = bstr.length
        // 根据解码结果字符串的长度创建一个等长的整形数字数组
        // 但在创建时 所有元素初始值都为 0
        const u8arr = new Uint8Array(n)
        // 将整形数组的每个元素填充为解码结果字符串对应位置字符的UTF-16 编码单元
        while (n--) {
            // charCodeAt()：获取给定索引处字符对应的 UTF-16 代码单元
            u8arr[n] = bstr.charCodeAt(n)
        }
        // 利用构造函数创建File文件对象
        // new File(bits, name, options)
        const file = new File([u8arr], `${fileName}.${suffix}`, {
            type: type
        })
        // 将File文件对象返回给方法的调用者
        return file
        },
        stopNavigator() {
            navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
            const videoElement = this.$refs.videoElement;
            videoElement.srcObject = stream;
            videoElement.srcObject.getTracks()[0].stop()
            console.log("摄像头关闭了")
          })
        },
        startRecord() {
			console.log("start to record now.");
            this.$message.success("正在录音...")
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
		stopRecord() {
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
					// xhr.open("POST","http://localhost:9090/file/uploadImg");
                    xhr.open("POST","http://localhost:8000/query");
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
            self.recorder.stop();
		},
        handlenext() {
            document.querySelector("#TrainingResults").click();
        },
        
    },
    created() 
    {
        this.msg=this.voiceMsg
        this.$ws.addEventListener('message', (event) => {
        // 处理 WebSocket 消息
        const message = event.data;
        console.log('WebSocket消息：', message);
        
        if (message === '1') {
            this.selected(0);
        } else if (message === '2') {
            this.selected(1);
        } else if (message === '3') {
            this.selected(2);
        } else if (message === '4') {
            this.selected(3);
        } else if (message === '5') {
            this.selected(4);
        } else if (message === '6') {
            this.selected(5);
        } else if (message === '7') {
            this.selected(6);
        } else if (message === '8') {
            this.selected(7);
        } else if (message === '9') {
            this.selected(8);
        } else if (message === '10') {
            this.selected(9);
        } else if (message.includes('向上滑动') && this.isactive >0) {
            this.selected(this.isactive-1);
        } else if (message.includes('向下滑动') && this.isactive <9) {
            this.selected(this.isactive+1);
        } else if (message === 'gg') {
            this.playVedio();
        } 
        
        })
        

    },
    mounted() {
    }

}
</script>

<style lang="scss" scoped >
.ulStyle{
    list-style: none;
    cursor: pointer;
}
.ulStyle>li{
    text-align: center;
    line-height:7vh; 
    border: 1px solid red;
    margin-top: 30%;
    font-size: 3vh;
    height: 7vh;
    width:4vw;
    border: 2px solid #dcdfe6;
    color: #036eff;
    border-radius: 10%;
}
.isActive{
    background-color:#036eff;
    color: #ffffff !important;
    border: none !important;
}
.acTived,.acTivedTwo{
    pointer-events: none;
}
.msg{
    margin-left: 2%;
}
.showIcon{

}
.isShow,.isshow{
    display: none;
}
.el-card{
    --el-card-padding: 0;
}
.vedio{
    width: 90%;
    height: 70vh;
    margin-bottom: 4vh;
}
.isShowVedio{
    display: none;
}
.el-icon{
    margin-left: 2%;
}
</style>