import router from '@/router/index.js';
import RecorderManager from '../xunfei/index.esm.js';
import CryptoJS from  "crypto-js";
import { ElMessage } from 'element-plus'
export default function () {
    const recorderManager = new RecorderManager("/worker");

    let iatWS;

    let resultText;

    const config = {
        appId: "1b151574",
        secretKey:"f8a908bd9dce68d84d7439c235194557"
    }

    recorderManager.onStart = ()=>{
        console.log("录音已开启");
        ElMessage({
            message: '录音已开启!现在可以和我语音对话了',
            type: 'success',
          })
    }

    function generateWebSocketUrl() {
        // 请求地址根据语种不同变化
        const url = "wss://rtasr.xfyun.cn/v1/ws";
        const {appId,secretKey} = config;
        const ts = Math.floor(new Date().getTime() / 1000);
        const signa = CryptoJS.MD5(appId + ts).toString();
        const signatureSha = CryptoJS.HmacSHA1(signa, secretKey);
        const signature = CryptoJS.enc.Base64.stringify(signatureSha);
        return `${url}?appid=${appId}&ts=${ts}&signa=${encodeURIComponent(signature)}`;
    }

    function connectWebSocket() {
        const websocketUrl = generateWebSocketUrl();
        if ("WebSocket" in window) {
            iatWS = new WebSocket(websocketUrl);
        } else if ("MozWebSocket" in window) {
            iatWS = new MozWebSocket(websocketUrl);
        } else {
            alert("浏览器不支持WebSocket");
            return;
        }
        console.log("正在连接WebSocket...");
        iatWS.onopen = (e) => {
            // 开始录音
            recorderManager.start({
                sampleRate: 16000,
                frameSize: 1280,
            });
        };
        iatWS.onmessage = (e) => {
            renderResult(e.data);
        };
        iatWS.onerror = (e) => {
            recorderManager.stop();
            console.error("socket连接失败",e)
        };
        iatWS.onclose = (e) => {
            recorderManager.stop();
            console.error("socket断开连接",e)
        };
    }

    function renderResult(resultData) {
        let jsonData = JSON.parse(resultData);
        if (jsonData.action === "started") {
            console.log("握手成功");
        } else if (jsonData.action === "result") {
            const data = JSON.parse(jsonData.data)
            console.log("message -> ",data)
            // 转写结果
            let resultTextTemp = ""
            data.cn.st.rt.forEach((j) => {
                j.ws.forEach((k) => {
                    k.cw.forEach((l) => {
                        resultTextTemp += l.w;
                    });
                });
            });
            
            console.log("最终结果 -> ",resultTextTemp);
            if(resultTextTemp.includes("中心")){
                console.log("个人中心---")
                router.push('/information')
            }else if(resultTextTemp.includes('通知') || resultTextTemp=='系统'){
                console.log("系统通知----")
                router.push('/notice')
            }else if(resultTextTemp.includes('缩放') || resultTextTemp=='系统'){
                console.log("进入缩放页面----")
                router.push('/change')
                // document.getElementById('system').click();
            }else if(resultTextTemp.includes('识别') || resultTextTemp=='验证'){
                console.log("识别验证----")
                router.push('/body')
            }else if(resultTextTemp.includes('首页') || resultTextTemp=='系统'){
                console.log("首页----")
                router.push('/')
            }else if(resultTextTemp.includes('伤情') && resultTextTemp.includes("识别")){
                console.log("伤情识别----")
                router.push("/homeTrain")
            }else if(resultTextTemp.includes('急救指南')){
                    console.log("急救指南----")
                    router.push("/study/right/detail")
            }else if(resultTextTemp.includes('教学模式')&& resultTextTemp.includes("模式")){
                    console.log("教学模式----")
                    router.push("/Teaching")  
            }else if(resultTextTemp.includes('肢体') && resultTextTemp.includes("动作")){
                    console.log("肢体动作识别----")
                    document.querySelector('#body').click()
            }else if(resultTextTemp.includes('手部') && resultTextTemp.includes("模型")){
                    console.log("手部模型识别----")
                    document.querySelector('#hand').click()
            



            }else if(resultTextTemp.includes('学习') && resultTextTemp.includes("模式")){
                console.log("学习模式----")
                router.push("/study/right/1")
                // document.getElementById("img1").click()
            }else if(resultTextTemp.includes('问答模式')){
                console.log("问答模式----")
                router.push("/train")
            }else if(resultTextTemp.includes('考核模式')){
                console.log("考核模式----")
                router.push("/examine")
            }else if(resultTextTemp.includes('开始') && resultTextTemp.includes("考核")){
                console.log("开始考核----")
                // router.push("/ExamineInner")
                document.querySelector("#ExamineInner").click()
            }else if(resultTextTemp.includes('详情')){
                console.log("查看详情----")
                router.push('/detail')
            }else if(resultTextTemp.includes('辨别烧伤')){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                document.querySelector('#leftItem1').click()
            }else if(resultTextTemp.includes("返回")){
                router.back()
            }else if(resultTextTemp.includes("烧伤") && resultTextTemp.includes("救治方案")){
                // router.push("/study/right/burn")
                document.querySelector('#leftItem2').click()
            }else if(resultTextTemp.includes("烧伤") && resultTextTemp.includes("药品")){
                // router.push("/study/right/burn")
                document.querySelector('#leftItem13').click()    

            }else if(resultTextTemp.includes("辨别骨折")){
                console.log("辨别骨折----")
                // router.push("/study/right/2")
                document.querySelector('#leftItem3').click()
            }else if(resultTextTemp.includes("骨折") && resultTextTemp.includes("救治方案")){
                console.log("骨折救治方案----")
            }else if(resultTextTemp.includes("骨折") && resultTextTemp.includes("药品")){
                console.log("骨折药品----")
                // router.push("/study/right/fracture")
                document.querySelector('#leftItem14').click()

            }else if(resultTextTemp.includes("辨别撞伤")){
                console.log("辨别撞伤----")
                // router.push("/study/right/3")
                document.querySelector('#leftItem5').click()
            }else if(resultTextTemp.includes("撞伤") & resultTextTemp.includes("救治方案")){
                console.log("撞伤救治方案------")
                // router.push("/study/right/bruise")
                document.querySelector('#leftItem6').click()
            }else if(resultTextTemp.includes("撞伤") & resultTextTemp.includes("药品")){
                console.log("撞伤救治方案------")
                // router.push("/study/right/bruise")
                document.querySelector('#leftItem15').click()

            }else if(resultTextTemp.includes("辨别划伤")){
                console.log("辨别划伤----")
                // router.push("/study/right/4")
                document.querySelector('#leftItem7').click()
            }else if(resultTextTemp.includes("划伤") & resultTextTemp.includes("救治方案")){
                console.log("划伤救治方案-----")     
                // router.push("/study/right/scratches")
                document.querySelector('#leftItem8').click()
            }else if(resultTextTemp.includes("划伤") & resultTextTemp.includes("药品")){
                console.log("划伤救治方案-----")
                // router.push("/study/right/scratches")
                document.querySelector('#leftItem16').click()

            }else if(resultTextTemp.includes("辨别枪弹伤")){
                console.log("辨别枪弹伤----")
                // router.push("/study/right/5")
                document.querySelector('#leftItem9').click()
            }else if(resultTextTemp.includes("枪弹伤") & resultTextTemp.includes("救治方案")){
                console.log("枪弹伤救治方案----")
                // router.push("/study/right/gunshot")
                document.querySelector('#leftItem10').click()
            }else if(resultTextTemp.includes("枪弹伤") & resultTextTemp.includes("药品")){
                console.log("枪弹伤救治方案----")
                // router.push("/study/right/gunshot")
                document.querySelector('#leftItem17').click()

            }else if(resultTextTemp.includes("辨别炸伤")){
                console.log("辨别炸伤----")
                // router.push("/study/right/6")
                document.querySelector('#leftItem11').click()
            }else if(resultTextTemp.includes("炸伤") & resultTextTemp.includes("救治方案")){
                console.log("炸伤救治方案-----")
                // router.push("/study/right/explosion")
                document.querySelector('#leftItem12').click()
            }else if(resultTextTemp.includes("炸伤") & resultTextTemp.includes("救治方案")){
                console.log("炸伤救治方案-----")
                // router.push("/study/right/explosion")
                document.querySelector('#leftItem12').click()
            }else if(resultTextTemp.includes("炸伤") & resultTextTemp.includes("药品")){
                console.log("炸伤救治方案-----")
                // router.push("/study/right/explosion")
                document.querySelector('#leftItem18').click()
            
            }else if(resultTextTemp.includes("推荐") & resultTextTemp.includes("救治")
            & resultTextTemp.includes("方案")){
                console.log("推荐救治方案-----")
                // router.push("/study/right/explosion")
                document.querySelector('#tjjzfa').click()
            

            }else if(resultTextTemp.includes("资料")){
                console.log("个人资料----")
                // router.push("/information")
                document.querySelector("#personaldata").click()
            }else if(resultTextTemp.includes("成绩")&resultTextTemp.includes("训练")){
                console.log("训练成绩----")
                // router.push("/trainingResults")
                document.querySelector("#TrainingResults").click()
            }else if(resultTextTemp.includes("进度") && resultTextTemp.includes("学习")){
                console.log("学习进度-----")
                // router.push("/studySchedule")
                document.querySelector("#studySchedule").click()

            }else if(resultTextTemp.includes("操作")|| resultTextTemp.includes("视频")){
                console.log("操作视频-----")
                document.querySelector('#myButton').click()
            }else if(resultTextTemp.includes("第一题")|| resultTextTemp.includes("第1题")){
                console.log("第一题-----")
                document.querySelector("#index1").click()
            }else if(resultTextTemp.includes("第二题")|| resultTextTemp.includes("第2题")){
                console.log("第一题-----")
                document.querySelector("#index2").click()
            }else if(resultTextTemp.includes("第三题")|| resultTextTemp.includes("第3题")){
                console.log("第一题-----")
                document.querySelector("#index3").click()
            }else if(resultTextTemp.includes("第四题")|| resultTextTemp.includes("第4题")){
                console.log("第一题-----")
                document.querySelector("#index4").click()
            }else if(resultTextTemp.includes("第五题")|| resultTextTemp.includes("第5题")){
                console.log("第一题-----")
                document.querySelector("#index5").click()
            }else if(resultTextTemp.includes("第六题")|| resultTextTemp.includes("第6题")){
                console.log("第一题-----")
                document.querySelector("#index6").click()
            }else if(resultTextTemp.includes("第七题")|| resultTextTemp.includes("第7题")){
                console.log("第一题-----")
                document.querySelector("#index7").click()
            }else if(resultTextTemp.includes("第八题")|| resultTextTemp.includes("第8题")){
                console.log("第一题-----")
                document.querySelector("#index8").click()
            }else if(resultTextTemp.includes("第九题")|| resultTextTemp.includes("第9题")){
                console.log("第一题-----")
                document.querySelector("#index9").click()
            }else if(resultTextTemp.includes("第十题")|| resultTextTemp.includes("第10题")){
                console.log("第一题-----")
                document.querySelector("#index10").click()
            }else if(resultTextTemp.includes("第十一题")|| resultTextTemp.includes("第11题")){
                console.log("第一题-----")
                document.querySelector("#index11").click()
            }else if(resultTextTemp.includes("第十二题")|| resultTextTemp.includes("第12题")){
                console.log("第一题-----")
                document.querySelector("#index12").click()
            }else if(resultTextTemp.includes("第十三题")|| resultTextTemp.includes("第13题")){
                console.log("第一题-----")
                document.querySelector("#index13").click()
            }else if(resultTextTemp.includes("第十四题")|| resultTextTemp.includes("第14题")){
                console.log("第一题-----")
                document.querySelector("#index14").click()
            }else if(resultTextTemp.includes("第十五题")|| resultTextTemp.includes("第15题")){
                console.log("第一题-----")
                document.querySelector("#index15").click()
            }else if(resultTextTemp.includes("第十六题")|| resultTextTemp.includes("第16题")){
                console.log("第一题-----")
                document.querySelector("#index16").click()
            }else if(resultTextTemp.includes("第十七题")|| resultTextTemp.includes("第17题")){
                console.log("第一题-----")
                document.querySelector("#index17").click()
            }else if(resultTextTemp.includes("第十八题")|| resultTextTemp.includes("第18题")){
                console.log("第一题-----")
                document.querySelector("#index18").click()
            }else if(resultTextTemp.includes("第十九题")|| resultTextTemp.includes("第19题")){
                console.log("第一题-----")
                document.querySelector("#index19").click()
            }else if(resultTextTemp.includes("第二十题")|| resultTextTemp.includes("第20题")){
                console.log("第一题-----")
                document.querySelector("#index20").click()
            }else if(resultTextTemp.includes("第二十一题")|| resultTextTemp.includes("第21题")){
                console.log("第一题-----")
                document.querySelector("#index21").click()
            }else if(resultTextTemp.includes("第二十二题")|| resultTextTemp.includes("第22题")){
                console.log("第一题-----")
                document.querySelector("#index22").click()
            }else if(resultTextTemp.includes("第二十三题")|| resultTextTemp.includes("第23题")){
                console.log("第一题-----")
                document.querySelector("#index23").click()
            }else if(resultTextTemp.includes("第二十四题")|| resultTextTemp.includes("第24题")){
                console.log("第一题-----")
                document.querySelector("#index24").click()
            }else if(resultTextTemp.includes("第二十五题")|| resultTextTemp.includes("第25题")){
                console.log("第一题-----")
                document.querySelector("#index25").click()
            }else if(resultTextTemp.includes("第二十六题")|| resultTextTemp.includes("第26题")){
                console.log("第一题-----")
                document.querySelector("#index26").click()
            }else if(resultTextTemp.includes("第二十七题")|| resultTextTemp.includes("第27题")){
                console.log("第一题-----")
                document.querySelector("#index27").click()
            }else if(resultTextTemp.includes("第二十八题")|| resultTextTemp.includes("第28题")){
                console.log("第一题-----")
                document.querySelector("#index28").click()
            }else if(resultTextTemp.includes("第二十九题")|| resultTextTemp.includes("第29题")){
                console.log("第一题-----")
                document.querySelector("#index29").click()
            }else if(resultTextTemp.includes("第三十题")|| resultTextTemp.includes("第30题")){
                console.log("第一题-----")
                document.querySelector("#index30").click()
            }else if(resultTextTemp.includes("这个是")|| resultTextTemp.includes("擦伤")){
                console.log("判断题-----")
                document.querySelector("#determine").click()
            }else if(resultTextTemp.includes("这道题")|| resultTextTemp.includes("选")){
                console.log("选择题-----")
                document.querySelector("#select").click()
            }


            if (data.cn.st.type === 0) {
                resultText += resultTextTemp;
                resultTextTemp = ""
            }
        } else if (jsonData.action === "error") {
            // 连接发生错误
            console.log("出错了:", jsonData);
        }
    }

    recorderManager.onFrameRecorded = ({ isLastFrame, frameBuffer }) => {
        if (iatWS.readyState === iatWS.OPEN) {
            iatWS.send(new Int8Array(frameBuffer));
            if (isLastFrame) {
                iatWS.send('{"end": true}');
                console.log("结束录音...");
            }
        }
    };
    recorderManager.onStop = () => {
        console.log("结束事件触发..")
    };
    connectWebSocket();
}
