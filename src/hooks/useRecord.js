import router from '@/router/index.js';
import RecorderManager from '../xunfei/index.esm.js';
import CryptoJS from  "crypto-js";
import { ElMessage } from 'element-plus'

function clickElement(selector) {
    let element = document.querySelector(selector);
    if (element) {
        element.click();
    } else {
        console.log(`元素${selector}不存在于当前页面。`);
    }
}


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
            // 页面跳转
            if(resultTextTemp.includes("中心")){
                console.log("个人中心---")
                router.push('/information')
            }else if(resultTextTemp.includes('通知') || resultTextTemp=='系统'){
                console.log("系统通知----")
                router.push('/notice')
            // }else if(resultTextTemp.includes('缩放') || resultTextTemp=='系统'){
            //     console.log("进入缩放页面----")
            //     router.push('/change')
                // document.getElementById('system').click();
            }else if(resultTextTemp.includes('伤情') && resultTextTemp.includes("识别")){
                console.log("伤情识别----")
                router.push("/homeTrain")
            }else if(resultTextTemp.includes('急救指南')){
                console.log("急救指南----")
                router.push("/study/right/detail")
            }else if(resultTextTemp.includes('教学模式')&& resultTextTemp.includes("模式")){
                console.log("教学模式----")
                router.push("/Teaching") 
            }else if(resultTextTemp.includes('首页') || resultTextTemp=='系统'){
                console.log("首页----")
                router.push('/')
                 
            }else if(resultTextTemp.includes('载入') ){
                    console.log("载入图片----")
                    document.querySelector('#zairu').click()
                    clickElement('#zairu')
            }
            else if(resultTextTemp.includes('打开')&& resultTextTemp.includes("拍摄")){
                    console.log("拍照进行识别")
                    // document.querySelector('#paishe').click()      
                    clickElement('#paishe') 
            }else if(resultTextTemp.includes('拍照')) {
                    console.log("拍照进行识别")
                    // document.querySelector('#paizhao').click()
                    clickElement('#paizhao') 
            }else if(resultTextTemp.includes('取消')&& resultTextTemp.includes("拍摄")){
                    console.log("取消拍摄")
                    // document.querySelector('#quxiao').click()
                    clickElement('#quxiao') 
            }else if(resultTextTemp.includes('提交')) {
                    console.log("提交照片")
                    // document.querySelector('#tijiao').click()
                    clickElement('#tijiao') 
            }else if(resultTextTemp.includes('第一级')&& resultTextTemp.includes('救治方案一')){
                    console.log("辨别烧伤----")
                    // router.push("/study/right/1")
                    // document.querySelector('#xxx1').click()
                    clickElement('#xxx1') 
            }else if(resultTextTemp.includes('第一级')&& resultTextTemp.includes('救治方案二')){
                    console.log("辨别烧伤----")
                    // router.push("/study/right/1")
                    // document.querySelector('#xxx2').click()
                    clickElement('#xxx2') 
            }else if(resultTextTemp.includes('第一级')&& resultTextTemp.includes('救治方案三')){
                    console.log("辨别烧伤----")
                    // router.push("/study/right/1")
                    // document.querySelector('#xxx3').click()
                    clickElement('#xxx3') 
            }else if(resultTextTemp.includes('第二级')&& resultTextTemp.includes('救治方案')){
                    console.log("辨别烧伤----")
                    // router.push("/study/right/1")
                    // document.querySelector('#xxx21').click()
                    clickElement('#xxx21') 
            }else if(resultTextTemp.includes('第三级')&& resultTextTemp.includes('救治方案')){
                    console.log("辨别烧伤----")
                    // router.push("/study/right/1")
                    // document.querySelector('#xxx31').click()
                    clickElement('#xxx31') 
            

            }else if(resultTextTemp.includes('学习') && resultTextTemp.includes("模式")){
                console.log("学习模式----")
                router.push("/study/right/1")
                // document.getElementById("img1").click()
            }else if(resultTextTemp.includes('做题模式')){
                console.log("做题模式----")
                router.push("/train")
            }else if(resultTextTemp.includes('考核模式')){
                console.log("考核模式----")
                router.push("/examine")
            }else if(resultTextTemp.includes('开始') && resultTextTemp.includes("考核")){
                console.log("开始考核----")
                // router.push("/ExamineInner")
                // document.querySelector("#ExamineInner").click()
                clickElement('#ExamineInner') 
           
            }else if(resultTextTemp.includes('等级一')&& resultTextTemp.includes('烧伤')&& resultTextTemp.includes("救治方案")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem1').click()
                clickElement('#leftItem1') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S11')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem1').click()
                clickElement('#leftItem1') 

            }else if(resultTextTemp.includes("返回")){
                router.back()
            }else if(resultTextTemp.includes('等级二')&&resultTextTemp.includes("烧伤") && resultTextTemp.includes("救治方案")){
                // router.push("/study/right/burn")
                // document.querySelector('#leftItem2').click()
                clickElement('#leftItem2') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S12')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem2').click()
                clickElement('#leftItem2') 

            // }else if(resultTextTemp.includes('等级三')&&resultTextTemp.includes("烧伤") && resultTextTemp.includes("救治方案")){
            //     // router.push("/study/right/burn")
            //     document.querySelector('#leftItem13').click()   
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S13')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem13').click() 
                clickElement('#leftItem3') 


            }else if(resultTextTemp.includes("等级一")&& resultTextTemp.includes("骨折") && resultTextTemp.includes("救治方案")){
                
                console.log("辨别骨折----")
                // router.push("/study/right/2")
                // document.querySelector('#leftItem3').click()
                clickElement('#leftItem3') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S21')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem3').click()
                clickElement('#leftItem3') 


            }else if(resultTextTemp.includes("等级二")&&resultTextTemp.includes("骨折") && resultTextTemp.includes("救治方案")){
                console.log("骨折救治方案----")
                // document.querySelector('#leftItem4').click()
                clickElement('#leftItem4') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S22')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem4').click()
                clickElement('#leftItem4') 

            
            // }else if(resultTextTemp.includes("等级三")&&resultTextTemp.includes("骨折") && resultTextTemp.includes("救治方案")){
            //     console.log("骨折药品----")
            //     // router.push("/study/right/fracture")
            //     document.querySelector('#leftItem14').click()
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S23')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem14').click()
                clickElement('#leftItem14') 


            }else if(resultTextTemp.includes("等级一")&& resultTextTemp.includes("撞伤") && resultTextTemp.includes("救治方案")){
                console.log("辨别撞伤----")
                // router.push("/study/right/3")
                // document.querySelector('#leftItem5').click()
                clickElement('#leftItem5') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S31')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem5').click()
                clickElement('#leftItem5') 



            }else if(resultTextTemp.includes("等级二")&&resultTextTemp.includes("撞伤") & resultTextTemp.includes("救治方案")){
                console.log("撞伤救治方案------")
                // router.push("/study/right/bruise")
                // document.querySelector('#leftItem6').click()
                clickElement('#leftItem6') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S32')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem6').click()
                clickElement('#leftItem6') 



            // }else if(resultTextTemp.includes("等级三")&&resultTextTemp.includes("撞伤") & resultTextTemp.includes("救治方案")){
            //     console.log("撞伤救治方案------")
            //     // router.push("/study/right/bruise")
            //     document.querySelector('#leftItem15').click()
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S33')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem15').click()
                clickElement('#leftItem15') 


            }else if(resultTextTemp.includes("等级一")&& (resultTextTemp.includes("划伤")||resultTextTemp.includes("华商")) && resultTextTemp.includes("救治方案")){
                console.log("辨别划伤----")
                // router.push("/study/right/4")
                // document.querySelector('#leftItem7').click()
                clickElement('#leftItem7') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('41')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem7').click()
                clickElement('#leftItem7') 

            
            }else if(resultTextTemp.includes("等级二")&& resultTextTemp.includes("划伤") && resultTextTemp.includes("救治方案")){
                console.log("划伤救治方案-----")     
                // router.push("/study/right/scratches")
                // document.querySelector('#leftItem8').click()
                clickElement('#leftItem8') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S42')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem8').click()
                clickElement('#leftItem8') 


            // }else if(resultTextTemp.includes("等级三")&& resultTextTemp.includes("划伤") && resultTextTemp.includes("救治方案")){
            //     console.log("划伤救治方案-----")
            //     // router.push("/study/right/scratches")
            //     document.querySelector('#leftItem16').click()
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S43')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem16').click()
                clickElement('#leftItem16') 


            }else if(resultTextTemp.includes("等级一")&& resultTextTemp.includes("枪弹伤") && resultTextTemp.includes("救治方案")){
                console.log("辨别枪弹伤----")
                // router.push("/study/right/5")
                // document.querySelector('#leftItem9').click()
                clickElement('#leftItem9') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S51')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem9').click()
                clickElement('#leftItem9') 

            }else if(resultTextTemp.includes("等级二")&& resultTextTemp.includes("枪弹伤") && resultTextTemp.includes("救治方案")){
                console.log("枪弹伤救治方案----")
                // router.push("/study/right/gunshot")
                // document.querySelector('#leftItem10').click()
                clickElement('#leftItem10') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S52')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem10').click()
                clickElement('#leftItem10') 


            // }else if(resultTextTemp.includes("等级三")&& resultTextTemp.includes("枪弹伤") && resultTextTemp.includes("救治方案")){
            //     console.log("枪弹伤救治方案----")
            //     // router.push("/study/right/gunshot")
            //     document.querySelector('#leftItem17').click()
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S53')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem17').click()
                clickElement('#leftItem17') 


            }else if(resultTextTemp.includes("等级一")&& resultTextTemp.includes("炸伤") && resultTextTemp.includes("救治方案")){
                console.log("辨别炸伤----")
                // router.push("/study/right/6")
                // document.querySelector('#leftItem11').click()
                clickElement('#leftItem11') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S61')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem11').click()
                clickElement('#leftItem11') 


            }else if(resultTextTemp.includes("等级二")&& resultTextTemp.includes("炸伤") && resultTextTemp.includes("救治方案")){
                console.log("炸伤救治方案-----")
                // router.push("/study/right/explosion")
                // document.querySelector('#leftItem12').click()
                clickElement('#leftItem12') 

            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S62')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem12').click()
                clickElement('#leftItem12') 


            // }else if(resultTextTemp.includes("等级三")&& resultTextTemp.includes("炸伤") && resultTextTemp.includes("救治方案")){
            //     console.log("炸伤救治方案-----")
            //     // router.push("/study/right/explosion")
            //     document.querySelector('#leftItem18').click()
            }else if(resultTextTemp.includes('进入')&& resultTextTemp.includes('S63')&& resultTextTemp.includes("节点")){
                console.log("辨别烧伤----")
                // router.push("/study/right/1")
                // document.querySelector('#leftItem18').click()
                clickElement('#leftItem18') 

            // }else if(resultTextTemp.includes("炸伤") & resultTextTemp.includes("药品")){
            //     console.log("炸伤救治方案-----")
            //     // router.push("/study/right/explosion")
            //     document.querySelector('#leftItem18').click()
            
            }else if(resultTextTemp.includes("推荐") & resultTextTemp.includes("救治")
            & resultTextTemp.includes("方案")){
                console.log("推荐救治方案-----")
                // router.push("/study/right/explosion")
                // document.querySelector('#tjjzfa').click()
                clickElement('#tjjzfa') 

            

            }else if(resultTextTemp.includes("资料")){
                console.log("个人资料----")
                // router.push("/information")
                // document.querySelector("#personaldata").click()
                clickElement('#personaldata') 

            // }else if(resultTextTemp.includes("成绩")&resultTextTemp.includes("训练")){
            //     console.log("训练成绩----")
            //     // router.push("/trainingResults")
            //     document.querySelector("#TrainingResults").click()
            // }else if(resultTextTemp.includes("进度") && resultTextTemp.includes("学习")){
            //     console.log("学习进度-----")
            //     // router.push("/studySchedule")
            //     document.querySelector("#studySchedule").click()

            // }else if(resultTextTemp.includes("操作")|| resultTextTemp.includes("视频")){
            //     console.log("操作视频-----")
            //     document.querySelector('#myButton').click()
            // }else if(resultTextTemp.includes("第一题")|| resultTextTemp.includes("第1题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index1").click()
            // }else if(resultTextTemp.includes("第二题")|| resultTextTemp.includes("第2题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index2").click()
            // }else if(resultTextTemp.includes("第三题")|| resultTextTemp.includes("第3题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index3").click()
            // }else if(resultTextTemp.includes("第四题")|| resultTextTemp.includes("第4题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index4").click()
            // }else if(resultTextTemp.includes("第五题")|| resultTextTemp.includes("第5题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index5").click()
            // }else if(resultTextTemp.includes("第六题")|| resultTextTemp.includes("第6题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index6").click()
            // }else if(resultTextTemp.includes("第七题")|| resultTextTemp.includes("第7题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index7").click()
            // }else if(resultTextTemp.includes("第八题")|| resultTextTemp.includes("第8题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index8").click()
            // }else if(resultTextTemp.includes("第九题")|| resultTextTemp.includes("第9题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index9").click()
            // }else if(resultTextTemp.includes("第十题")|| resultTextTemp.includes("第10题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index10").click()
            // }else if(resultTextTemp.includes("第十一题")|| resultTextTemp.includes("第11题")){
            //     console.log("第一题-----")
            //     document.querySelector("#index11").click()
            // }else if(resultTextTemp.includes("这个是") && resultTextTemp.includes("擦伤")){
            //     console.log("判断题-----")
            //     document.querySelector("#determine").click()
            // }else if(resultTextTemp.includes("这道题")&& resultTextTemp.includes("选")){
            //     console.log("选择题-----")
            //     document.querySelector("#select").click()
            // }

            // function clickElement(selector) {
            //     document.querySelector(selector).click();
            // }
            
        } else if(resultTextTemp.includes("操作") || resultTextTemp.includes("视频")) {
                console.log("操作视频-----");
                clickElement('#myButton');
            } else if(resultTextTemp.includes("第一题") || resultTextTemp.includes("第1题")) {
                console.log("第一题-----");
                clickElement("#index1");
            } else if(resultTextTemp.includes("第二题") || resultTextTemp.includes("第2题")) {
                console.log("第二题-----");
                clickElement("#index2");
            } else if(resultTextTemp.includes("第三题") || resultTextTemp.includes("第3题")) {
                console.log("第三题-----");
                clickElement("#index3");
            } else if(resultTextTemp.includes("第四题") || resultTextTemp.includes("第4题")) {
                console.log("第四题-----");
                clickElement("#index4");
            } else if(resultTextTemp.includes("第五题") || resultTextTemp.includes("第5题")) {
                console.log("第五题-----");
                clickElement("#index5");
            } else if(resultTextTemp.includes("第六题") || resultTextTemp.includes("第6题")) {
                console.log("第六题-----");
                clickElement("#index6");
            } else if(resultTextTemp.includes("第七题") || resultTextTemp.includes("第7题")) {
                console.log("第七题-----");
                clickElement("#index7");
            } else if(resultTextTemp.includes("第八题") || resultTextTemp.includes("第8题")) {
                console.log("第八题-----");
                clickElement("#index8");
            } else if(resultTextTemp.includes("第九题") || resultTextTemp.includes("第9题")) {
                console.log("第九题-----");
                clickElement("#index9");
            } else if(resultTextTemp.includes("第十题") || resultTextTemp.includes("第10题")) {
                console.log("第十题-----");
                clickElement("#index10");
            } else if(resultTextTemp.includes("第十一题") || resultTextTemp.includes("第11题")) {
                console.log("第十一题-----");
                clickElement("#index11");
            } else if(resultTextTemp.includes("这个是") && resultTextTemp.includes("擦伤")) {
                console.log("判断题-----");
                clickElement("#determine");
            } else if(resultTextTemp.includes("这道题") && resultTextTemp.includes("选")) {
                console.log("选择题-----");
                clickElement("#select");
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
