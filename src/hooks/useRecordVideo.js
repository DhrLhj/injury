import {nextTick} from 'vue';
import { ElMessage } from 'element-plus'
import router from '@/router/index.js';








let mediaRecorder;

let ws;
let flag = true;

// const websocketUrl = "ws://127.0.0.1:8000/room/123/";
// // if ("WebSocket" in window) {
// //     ws = new WebSocket(websocketUrl);
// // } else if ("MozWebSocket" in window) {
// //     ws = new MozWebSocket(websocketUrl);
// // } else {
// //     alert("浏览器不支持WebSocket");
// //     return;
// // }
// const WebSocketPlugin = {
//     install: (app) => {
//         if ("WebSocket" in window) {
//             ws = new WebSocket(websocketUrl);
//             } else if ("MozWebSocket" in window) {
//             ws = new MozWebSocket(websocketUrl);
//             } else {
//             alert("浏览器不支持WebSocket");
//             return;
//         }


//         ws.onopen = (e) => {
//             console.log("websocket连接成功");
//             ElMessage({
//                 message: '视频已开启!现在可以用手势操控了',
//                 type: 'success',
//             })
//             nextTick(async () => {
//                 await startRecording();
//             }).then(res =>{})
//         }
        
//         ws.onmessage = (e) => {
//         // console.log("message:=====",e.data);
//         // flag = true
        

//         if (e.data==='1024'){
//             console.log('qqq')
//             router.push("/home")
//         }
//         ws.send("start recognization");
//         //     console.log("进入第二题")
//         //     const hash = window.location.hash;
        
//         //     console.log(hash)
//         //     if(hash.includes('Teaching')){
//         //         router.push("/train")
//         //     }
//         //     console.log(hash)
//         //     if(hash.includes('home')){
//         //         router.push("/study/right/detail")
//         //     }
//         //     // else if(hash.includes('train')){
//         //     //     document.querySelector("#index1").click()
//         //     // }
//         // }else if(e.data==='3'){
//         //     console.log("进入第三题")
//         //     const hash = window.location.hash;
//         //     console.log(hash)
//         //     if(hash.includes('Teaching')){
//         //         router.push("/examine")
//         //     }
//         //     if(hash.includes('home')){
//         //         router.push("/Teaching")
//         //     }

//         // // 这里是一些特定手势

        
//         // }else if (e.data === 'good') {
//         //     console.log("返回首页");
//         //     router.push("/home");
//         // }else if (e.data === 'aa') {
//         //     console.log("进入个人中心");
//         //     router.push("/information");
//         // }else if (e.data === 'bad') {
//         //     console.log("查看系统通知");
//         //     router.push("/notice");
//         // }else if (e.data === 'bb') {
//         //     console.log("返回伤情识别");
//         //     router.push("/homeTrain");
//         // }else if (e.data === 'cc') {
//         //     console.log("返回急救指南");
//         //     router.push("/study/right/detail");
//         // }else if (e.data === 'dd') {
//         //     console.log("返回教学模式");
//         //     router.push("/Teaching");
//         // }else if (e.data === 'ee') {
//         //     console.log("进入烧伤救治方案");
//         //     router.push("/study/right/burn");
//         // }else if (e.data === 'ff') {
//         //     console.log("进入骨折救治方案");
//         //     router.push("/study/right/fracture");
//         // }else if (e.data === 'gg') {
//         //     console.log("进入撞伤救治方案");
//         //     router.push("/study/right/bruise");
//         // }else if (e.data === 'hh') {
//         //     console.log("进入划伤救治方案");
//         //     router.push("/study/right/scratches");
//         // }else if (e.data === 'ii') {
//         //     console.log("进入枪弹伤救治方案");
//         //     router.push("/study/right/gunshot");
//         // }else if (e.data === 'jj') {
//         //     console.log("进入炸伤救治方案");
//         //     router.push("/study/right/explosion");

//         //     // else if(hash.includes('train')){
//         //     //     document.querySelector("#index1").click()
//         //     // }
//         // }else if(e.data.includes("向下")){
//         //     const hash = window.location.hash
//         //     console.log(hash,'+++++++++++++++')
//         //     // if(hash.includes("information")){
//         //     //     document.querySelector("#TrainingResults").click()
//         //     // }else 
//         //     if(hash.includes("trainingResults")){
//         //         document.querySelector("#studySchedule").click()
//         //     }else if(hash.includes("studySchedule")){
//         //         document.querySelector("#information").click()
//         //     }else if(hash.includes("study/right/1")){
//         //         document.querySelector('#leftItem2').click()
//         //     }else if(hash.includes("study/right/burn")){
//         //         document.querySelector('#leftItem3').click()
//         //     }else if(hash.includes("study/right/2")){
//         //         document.querySelector('#leftItem4').click()
//         //     }else if(hash.includes("study/right/fracture")){
//         //         document.querySelector('#leftItem5').click()
//         //     }else if(hash.includes("study/right/3")){
//         //         document.querySelector('#leftItem6').click()
//         //     }else if(hash.includes("study/right/bruise")){
//         //         document.querySelector('#leftItem7').click()
//         //     }else if(hash.includes("study/right/4")){
//         //         document.querySelector('#leftItem8').click()
//         //     }else if(hash.includes("study/right/scratches")){
//         //         document.querySelector('#leftItem9').click()
//         //     }else if(hash.includes("study/right/5")){
//         //         document.querySelector('#leftItem10').click()
//         //     }else if(hash.includes("study/right/gunshot")){
//         //         document.querySelector('#leftItem11').click()
//         //     }else if(hash.includes("study/right/6")){
//         //         document.querySelector('#leftItem12').click()
//         //     }
//         // }else if(e.data.includes("向上")){
//         //     const hash = window.location.hash
//         //     console.log(hash,'+++++++++++++++')
//         //     // if(hash.includes("information")){
//         //     //     document.querySelector("#studySchedule").click()
//         //     // }else if(hash.includes("trainingResults")){
//         //     //     document.querySelector("#information").click()
//         //     // }else if(hash.includes("studySchedule")){
//         //     //     document.querySelector("#trainingResults").click()
//         //     // }
//         // }else if(e.data.includes("向右")){
//         //     const hash = window.location.hash
//         //     // if(hash.includes("information") || hash.includes("notice") || hash.includes("study/right/1") || hash.includes("train")){
//         //     //     router.back()
//         //     // }
//         // // }else if(e.data.includes("向左")){
//         // //     const hash = window.location.hash
//         // //     if(hash.includes("information") || hash.includes("notice") || hash.includes("study/right/1") || hash.includes("train")){
//         // //         router.back()
//         // //     }
//         // }else if(e.data==='4'){
//         //     document.querySelector("#index4").click()
//         // // }else if(e.data==='5'){
//         // //     document.querySelector("#index5").click()
//         // }else if(e.data==='6'){
//         //     document.querySelector("#index6").click()
//         // }else if(e.data==='7'){
//         //     document.querySelector("#index7").click()
//         // }else if(e.data==='8'){
//         //     document.querySelector("#index8").click()
//         // }else if(e.data==='9'){
//         //     document.querySelector("#index9").click()
//         // }else if(e.data==='10'){
//         //     document.querySelector("#index10").click()
//         // }

//         }
//         ws.onerror = (e) => {
//         console.error("socket连接失败",e);
//         }
//         ws.onclose = (e) => {
//         // console.log('websocket 断开: ' + e.code + ' ' + e.reason + ' ' + e.wasClean)
//         console.error("socket断开连接",e)
//         };

//         app.config.globalProperties.$ws = ws; 
//     }
// };
// async function startRecording() {
//     try {
//         // // const camera = document.getElementById('camera');
//         // const stream = await navigator.mediaDevices.getUserMedia({ video: true,audio:true });
//         // mediaRecorder = new MediaRecorder(stream);
//         // // camera.srcObject = stream;
//         // mediaRecorder.ondataavailable = (event) => {
//         //     if (event.data.size > 0 && flag) {
//         //         const blob = new Blob([event.data], { type: 'video/webm' });
//         //         ws.send(blob);
//         //         flag = false;
//         //     }
//         // };

//         // // 多久记录一次
//         // mediaRecorder.start(3000);

//         ws.send("start recognization");
//     } catch (error) {
//         console.error('recording error:', error);
//     }
// }

// function stopRecording() {
//     mediaRecorder.stop();
// }


    
// ;

// export default WebSocketPlugin;
// // export {WebSocketPlugin, stopRecording };


// 当按下任何键时执行此函数
function handleKeyDown(event) {
    if (event.key === '1' && event.key === 'p') {
        console.log('按下了 Enter 键');
    }
}

// 添加全局键盘响应事件监听器
document.addEventListener('keydown', handleKeyDown);
