import { onMounted, onBeforeUnmount } from 'vue';
import router from '@/router/index.js';

export default function useKey() {
    // 用于跟踪上一个按下的键
    let lastKeyPressed = null;

    onMounted(() => {
            document.addEventListener('keydown', handleKeyDown);
    });

    onBeforeUnmount(() => {
        document.removeEventListener('keydown', handleKeyDown);
    });

    function handleKeyDown(event) {
        if (event.key === '9') {
            console.log('按下了 9 键');
            lastKeyPressed = '9';
        } else if (event.key === 'm' && lastKeyPressed === '9') {
            console.log('先后按下了 9 和 m 键');
            router.push("/home");
            lastKeyPressed = null;  // 重置状态
        } else if (event.key === 'n' && lastKeyPressed === '9') {
            console.log('先后按下了 9 和 n 键');
            router.push("/homeTrain");
            lastKeyPressed = null;  // 重置状态
        } else if (event.key === 'l' && lastKeyPressed === '9') {
            console.log('先后按下了 9 和 l 键');
            router.push("/study/right/detail");
            lastKeyPressed = null;  // 重置状态
        } else {
            // 如果按下了其他键，重置状态
            lastKeyPressed = null;
        }
    }

    return {}
}
