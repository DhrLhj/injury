<template>
  <div class="main">
    <div id="myGraph" style="height:calc(100vh - 50px);">
      <RelationGraph ref="seeksRelationGraph" :options="graphOptions" :on-node-click="onNodeClick"
                     :on-line-click="onLineClick"/>
    </div>
  </div>
  <!-- <div id="imageModal" class="modal">
  <div class="modal-content">
    <img src="require('@/assets/imagebox/burn/1.jpg')" alt="Image">
  </div> -->
<!-- </div> -->
</template>

<script>
import router from '@/router/index.js';
import {ref, h} from 'vue';
import RelationGraph from 'relation-graph/vue3'
import {ElNotification} from "element-plus";

export default {
  name: "RightDetail",
  components: {RelationGraph},
  data() {
    return {
      graphOptions: {
        allowSwitchLineShape: true,
        allowSwitchJunctionPoint: true,
        defaultJunctionPoint: 'border',
      },
      myGraphPanelSize: {width: 1080, height: 600}, // 控制图谱大小
      graphInfo: {}, // 知识图谱信息
      center_: {x: -200, y: -100}, // 坐标轴起点
      currentChooseNodeId: "0", // 当前选中的nodeId
      keysPressed: {},
    }
  },

  methods: {


    
    showSeeksGraph() {
      const __graph_json_data = {
        "rootId": "a",
        "nodes": [
          {
            "id": "0",
            "text": "0",
            "color": "orange",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x,
            y: this.center_.y
          },// 根节点
          {
            "id": "s1",
            "text": "s1",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x,
            y: this.center_.y - 300
          },// s1
          {
            "id": "s2",
            "text": "s2",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y - 180
          },// s2
          {
            "id": "s3",
            "text": "s3",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y + 180
          },// s3
          {
            "id": "s4",
            "text": "s4",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x,
            y: this.center_.y + 300
          },// s4
          {
            "id": "s5",
            "text": "s5",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y + 180
          },// s5
          {
            "id": "s6",
            "text": "s6",
            "color": "#6495ED",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y - 180
          },// s6
          // s1
          {
            "id": "s11",
            "text": "s11",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 150,
            y: this.center_.y - 300
          },
          {
            "id": "s12",
            "text": "s12",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x,
            y: this.center_.y - 450
          },
          {
            "id": "s13",
            "text": "s13",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 150,
            y: this.center_.y - 300
          },
          //s2
          {
            "id": "s21",
            "text": "s21",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y - 330
          },
          {
            "id": "s22",
            "text": "s22",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 430,
            y: this.center_.y - 180
          },
          {
            "id": "s23",
            "text": "s23",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y - 45
          },
          //s3
          {
            "id": "s31",
            "text": "s31",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y + 45
          },
          {
            "id": "s32",
            "text": "s32",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 430,
            y: this.center_.y + 180
          },
          {
            "id": "s33",
            "text": "s33",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 280,
            y: this.center_.y + 330
          },
          //s4
          {
            "id": "s41",
            "text": "s41",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 150,
            y: this.center_.y + 300
          },
          {
            "id": "s42",
            "text": "s42",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x,
            y: this.center_.y + 450
          },
          {
            "id": "s43",
            "text": "s43",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x + 150,
            y: this.center_.y + 300
          },
          //s5
          {
            "id": "s51",
            "text": "s51",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y + 45
          },
          {
            "id": "s52",
            "text": "s52",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 430,
            y: this.center_.y + 180
          },
          {
            "id": "s53",
            "text": "s53",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y + 330
          },
          //s6
          {
            "id": "s61",
            "text": "s61",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y - 330
          },
          {
            "id": "s62",
            "text": "s62",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 430,
            y: this.center_.y - 180
          },
          {
            "id": "s63",
            "text": "s63",
            "opacity": 1,
            "disableDrag": true,
            "fixed": true,
            x: this.center_.x - 280,
            y: this.center_.y - 45
          },
        ],
        "lines": [
          {"from": "0", "to": "s1"},
          {"from": "0", "to": "s2"},
          {"from": "0", "to": "s3"},
          {"from": "0", "to": "s4"},
          {"from": "0", "to": "s5"},
          {"from": "0", "to": "s6"},
          // s1
          {"from": "s1", "to": "s11"},
          {"from": "s1", "to": "s12"},
          {"from": "s1", "to": "s13"},
          // s2
          {"from": "s2", "to": "s21"},
          {"from": "s2", "to": "s22"},
          {"from": "s2", "to": "s23"},
          // s3
          {"from": "s3", "to": "s31"},
          {"from": "s3", "to": "s32"},
          {"from": "s3", "to": "s33"},
          // s4
          {"from": "s4", "to": "s41"},
          {"from": "s4", "to": "s42"},
          {"from": "s4", "to": "s43"},
          // s5
          {"from": "s5", "to": "s51"},
          {"from": "s5", "to": "s52"},
          {"from": "s5", "to": "s53"},
          // s6
          {"from": "s6", "to": "s61"},
          {"from": "s6", "to": "s62"},
          {"from": "s6", "to": "s63"},
        ]
      }
      this.graphInfo = __graph_json_data
      // 以上数据中的node和link可以参考"Node节点"和"Link关系"中的参数进行配置
      this.$refs.seeksRelationGraph.setJsonData(__graph_json_data, (seeksRGGraph) => {
        // Called when the relation-graph is completed
      })
    },
    // 展示当前选中的节点
    onNodeClick(node, $event) {
      // let node = this.$refs.seeksRelationGraph.getNodeById(this.currentChooseNodeId) // 选择当前节点

      console.log('选中: ', node.id, node.x, node.y)
      console.log(node.targetNodes)
      // 在节点点击事件处理程序中执行跳转操作
      if (node.id === "s1") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能被烧伤，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的烧伤救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }
    if (node.id === "s2") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能会骨折，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的骨折救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }
    if (node.id === "s3") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能被撞伤，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的撞伤救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }if (node.id === "s4") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能被划伤，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的划伤救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }if (node.id === "s5") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能会中弹，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的枪弹伤救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }
    if (node.id === "s6") {
    // 如果节点的 id 为 "s1"，弹出提示框
       window.alert("在战场环境下，士兵可能被炸伤，按照救援难度分为三个等级：等级1、等级2、等级3，选择子节点进入不同等级的炸伤救治方案。");
      //  $("#imageModal").dialog({
      //   modal: true,
      //   width: "auto",
      //   height: "auto",
      // })
    }
      if (node.id ==="s11"){
        router.push("/study/right/1");
      }
      if (node.id ==="s12"){
        router.push("/study/right/burn");
      }
        if (node.id ==="s13"){
        router.push("/study/right/burnDrug"); }
        if (node.id ==="s21"){
        router.push("/study/right/2"); }
        if (node.id ==="s22"){
        router.push("/study/right/fracture"); }
        if (node.id ==="s23"){
        router.push("/study/right/fractureDrug"); }
        if (node.id ==="s31"){
        router.push("/study/right/3"); }
        if (node.id ==="s32"){
        router.push("/study/right/bruise"); }
        if (node.id ==="s33"){ 
        router.push("/study/right/bruiseDrug"); }
        if (node.id ==="s41"){
        router.push("/study/right/4"); }
        if (node.id ==="s42"){
        router.push("/study/right/scratches"); }
        if (node.id ==="s43"){
        router.push("/study/right/scratchesDrug"); }
        if (node.id ==="s51"){
        router.push("/study/right/5"); }
        if (node.id ==="s52"){
        router.push("/study/right/gunshot");
      }
      if (node.id ==="s53"){
        router.push("/study/right/gunshotDrug");
      }
      if (node.id ==="s61"){
        router.push("/study/right/6");
      }
      if (node.id ==="s62"){
        router.push("/study/right/explosion");
      }
      if (node.id ==="s63"){
        router.push("/study/right/explosionDrug");
      }

    },
        // 处理根据手势缩放事件
    handleZoomGraph(eventId) {
      // 获取要模拟滚动的元素
      let elements = document.querySelectorAll('span.c-mb-text');

      if (eventId === '9') { //放大
        for (let i = 0; i < elements.length; i++) {
          let element = elements[i];
          // 检查元素的文本内容是否为"放大"
          if (element.textContent === '放大') {
            element.click();
          }
        }
      } else if (eventId === '10') { // 缩小
        for (let i = 0; i < elements.length; i++) {
          let element = elements[i];
          // 检查元素的文本内容是否为"缩小"
          if (element.textContent === '缩小') {
            element.click();
          }
        }
      }
    },
    // 处理根据手势选择节点事件
    handleChooseNode(eventId) {
      console.log(this.currentChooseNodeId)
      let currentNode = this.$refs.seeksRelationGraph.getNodeById(this.currentChooseNodeId)
      let nodes = currentNode.targetNodes
      for (let i = 0; i < nodes.length; i++) {
        if (eventId === '1') {                  //上
          if (nodes[i].y < currentNode.y && nodes[i].x === currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '5') {           //下
          if (nodes[i].y > currentNode.y && nodes[i].x === currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '7') {           //左
          if (nodes[i].x < currentNode.x && nodes[i].y === currentNode.y) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '3') {           //右
          if (nodes[i].x > currentNode.x && nodes[i].y === currentNode.y) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '8') {           //左上
          if (nodes[i].y < currentNode.y && nodes[i].x < currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '2') {           //右上
          if (nodes[i].y < currentNode.y && nodes[i].x > currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '6') {           //左下
          if (nodes[i].y > currentNode.y && nodes[i].x < currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        } else if (eventId === '4') {           //右下
          if (nodes[i].y > currentNode.y && nodes[i].x > currentNode.x) {
            this.currentChooseNodeId = nodes[i].id
            break
          }
        }
      }
      this.$refs.seeksRelationGraph.focusNodeById(this.currentChooseNodeId)
    },


    // 展示当前选中的线
    onLineClick(lineObject, $event) {
      console.log('onLineClick:', lineObject)
    },
    // 处理点击节点事件
    handleClickNode(eventId) {
      let currentNode = this.$refs.seeksRelationGraph.getNodeById(this.currentChooseNodeId) // 选择当前节点
      // todo 执行接下来要做的事情
        
    },
    // 处理弹窗事件
    handleNotification(eventId) {
      // todo 使用element-plus的弹窗，内部内容根据需要修改
      ElNotification({
        title: 'Title',
        message: h('i', {style: 'color: teal'}, 'This is a reminder'),
        duration: 20000 // 弹窗持续时间20s
      })
    },
    handleGraphClick() {
      // 获取 RelationGraph 组件的引用
      const graphComponent = this.$refs.seeksRelationGraph;

      // 获取鼠标点击的目标元素
      const targetElement = graphComponent.$el.querySelector('.node');

      // 创建鼠标点击事件
      const event = new MouseEvent('click', {
        bubbles: true, // 是否冒泡
        cancelable: true, // 是否可取消
        view: window // 关联的视图对象
      });

      // 触发鼠标点击事件
      targetElement.dispatchEvent(event);
    },

    handleKeydown(event) {
      this.keysPressed[event.key] = true;
      if (this.keysPressed['d'] && this.keysPressed['e']) {
        this.onNodeClick(this.currentChooseNodeId)
        // this.handleGraphClick()
        this.keysPressed = {};
      }
    },
    handleKeyup(event) {
      delete this.keysPressed[event.key];
    },

    },

  mounted() {
    let self = this
    // 设置知识图谱并展示
    this.showSeeksGraph()
    // setTimeout(function () {
    //   self.handleChooseNode('6')
    //   self.handleNotification()
    // }, 0)
    // setTimeout(function () {
    //   self.handleChooseNode('1')
    //   self.handleNotification()
    // }, 2000)
    // setTimeout(function () {
    //   self.handleChooseNode('5')
    //   self.handleNotification()
    // }, 4000)
    // setTimeout(function () {
    //   self.handleChooseNode('2')
    //   self.handleNotification()
    // }, 6000)
    // setTimeout(function () {
    //   self.handleChooseNode('4')
    //   self.handleNotification()
    // }, 8000)
    // setTimeout(function () {
    //   self.handleChooseNode('3')
    //   self.handleNotification()
    // }, 10000)
    // setTimeout(function () {
    //   self.handleZoomGraph('10')
    // }, 0)
    // setTimeout(function () {
    //   self.handleZoomGraph('10')
    // }, 2000)
    // setTimeout(function () {
    //   self.handleZoomGraph('10')
    // }, 4000)
    // setTimeout(function () {
    //   self.handleZoomGraph('10')
    // }, 6000)

    document.addEventListener('keydown', this.handleKeydown);
    document.addEventListener('keyup', this.handleKeyup);
  },
  created() {
    let self = this
    this.handleWebSocketMessage=(event)=> {
      // 处理 WebSocket 消息
      const message = event.data;


      if (message !== "-1"){
        console.log('WebSocket消息：', message);
        if (message.includes('30')) {//方案选择下滑
          self.handleChooseNode('5')
          self.handleNotification()
        } else if (message.includes('29')) {//方案选择上滑
          self.handleChooseNode('1')
          self.handleNotification()
        } else if (message==='31') {//方案选择左滑
          self.handleChooseNode('7')
          self.handleNotification()
        } else if (message.includes('32')) {//方案选择右滑
          self.handleChooseNode('3')
          self.handleNotification()
        } else if (message.includes('33')) {//方案选择左上滑
          self.handleChooseNode('8')
          self.handleNotification()
        } else if (message.includes('34')) {//方案选择右上滑
          self.handleChooseNode('2')
          self.handleNotification()
        } else if (message.includes('35')) {//方案选择左下滑
          self.handleChooseNode('6')
          self.handleNotification()
        } else if (message.includes('36')) {//方案选择右下滑
          self.handleChooseNode('4')
          self.handleNotification()
        } else if (message.includes('19')) {//方案选择 手势7
          self.handleGraphClick()
        }
      }

    };
    // this.$ws.addEventListener('message', this.handleWebSocketMessage);

  },
  beforeUnmount() {
    console.log("退出detail")
    this.$ws.removeEventListener('message', this.handleWebSocketMessage);
    document.removeEventListener('keydown', this.handleKeydown);
    document.removeEventListener('keyup', this.handleKeyup);
  },

  }


</script>

<style>
.el-carousel__container {
  height: 100%;
}
</style>

