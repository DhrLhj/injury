import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'home',
    redirect:'/home',
    component: () => import( '../views/Manage.vue'),
    
    children:[
      { path:'/Teaching',name:'Teaching',component:()=>import('../views/Teaching.vue')},
      { path:'/home',name:'home',component:()=>import('../views/Home.vue')},
      { path:'/homeTrain',name:'homeTrain',component:()=>import('../views/HomeTrain.vue')},
      { path:'/train',name:'Train',component:()=>import('../views/Train.vue')},
      { path:'/record',name:'Record',component:()=>import('../components/Record.vue')},
      { path:'/index',name:'Index',component:()=>import('../views/Index.vue')},
      { path:'/test',name:'Test',component:()=>import('../views/Test.vue')},
      { path:'/examine',name:'Examine',component:()=>import('../views/Examine.vue')},
      { path: '/notice',name: 'notice',component: () => import('../views/Notice.vue')},
      { path:'/ExamineInner',name:'ExamineInner',component:()=>import('../views/ExamineInner.vue')},
      // { path:'/change',name:'change',component:()=>import('../views/change.vue')},
      { path:'/body',name:'body',component:()=>import('../views/body.vue')},
    ]
  },
  { path:'/study',
    name:'Study',
    component:()=>import('../views/Study.vue'),
    children:[
    { path:'/study/right/detail',name:'RightDetail',component:()=>import('../views/StudyVideo/RightDetail.vue')},
    { path:'/study/right/1',name:'RightVideoOne',component:()=>import('../views/StudyVideo/RightVideoOne.vue')},
    { path:'/study/right/2',name:'RightVideoTwo',component:()=>import('../views/StudyVideo/RightVideoTwo.vue')},
    { path:'/study/right/3',name:'RightVideoThree',component:()=>import('../views/StudyVideo/RightVideoThree.vue')},
    { path:'/study/right/4',name:'RightVideoFour',component:()=>import('../views/StudyVideo/RightVideoFour.vue')},
    { path:'/study/right/5',name:'RightVideoFive',component:()=>import('../views/StudyVideo/RightVideoFive.vue')},
    { path:'/study/right/6',name:'RightVideoSix',component:()=>import('../views/StudyVideo/RightVideoSix.vue')},
    { path:'/study/right/burn',name:'Burn',component:()=>import('../views/StudyPhoto/Burn.vue')},
    { path:'/study/right/burnDrug',name:'burnDrug',component:()=>import('../views/StudyPhoto/burnDrug.vue')},
    { path:'/study/right/bruise',name:'Bruise',component:()=>import('../views/StudyPhoto/Bruise.vue')},

    { path:'/study/right/bruiseDrug',name:'bruiseDrug',component:()=>import('../views/StudyPhoto/bruiseDrug.vue')},
    { path:'/study/right/fracture',name:'Fracture',component:()=>import('../views/StudyPhoto/Fracture.vue')},

    { path:'/study/right/fractureDrug',name:'fractureDrug',component:()=>import('../views/StudyPhoto/fractureDrug.vue')},
    { path:'/study/right/explosion',name:'Explosion',component:()=>import('../views/StudyPhoto/Explosion.vue')},

    { path:'/study/right/explosionDrug',name:'explosionDrug',component:()=>import('../views/StudyPhoto/explosionDrug.vue')},
    { path:'/study/right/gunshot',name:'Gunshot',component:()=>import('../views/StudyPhoto/Gunshot.vue')},

    { path:'/study/right/gunshotDrug',name:'gunshotDrug',component:()=>import('../views/StudyPhoto/gunshotDrug.vue')},
    { path:'/study/right/scratches',name:'Scratches',component:()=>import('../views/StudyPhoto/Scratches.vue')},
    
    { path:'/study/right/scratchesDrug',name:'scratchesDrug',component:()=>import('../views/StudyPhoto/scratchesDrug.vue')},
    { path:'/detail',name:'Detail',component:()=>import('../views//StudyPhoto/Detail.vue')},
  ]
  },
  { path:'/personal',
    name:'Personal',
    component:()=>import('../views/Personal.vue'),
    children:[
      { path:'/information',name:'Information',component:()=>import('../views/Information.vue')},
      { path:'/trainingResults',name:'TrainingResults',component:()=>import('../views/TrainingResults.vue')},
      { path:'/studySchedule',name:'StudySchedule',component:()=>import('../views/StudySchedule.vue')},
 
    ]
  },
  
  {
    path: '/left',
    name: 'left',
    component: () => import( '../views/Left.vue'),
    children:[
      // { path:'/home',name:'Home',component:()=>import('../views/Home.vue')},
  ]
},
{
  path:"/webscam",
  name:'webscam',
  component:()=>import('../views/webcam/Webcam.vue'),
}

]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  if (from.matched.length > 0 && from.matched[0].instances.default && typeof from.matched[0].instances.default.removeListeners === 'function') {
     from.matched[0].instances.default.removeListeners(); 
  }
  next()
})

export default router
