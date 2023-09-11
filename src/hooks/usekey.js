import { onMounted, onBeforeUnmount } from 'vue';

export default function useRecordVideo() {
    // ... 其他代码

    onMounted(() => {
        document.addEventListener('keydown', handleKeyDown);
    });

    onBeforeUnmount(() => {
        document.removeEventListener('keydown', handleKeyDown);
    });

    function handleKeyDown(event) {
        if (event.key === '9' && event.key === 'm') {
            console.log('按下了 Enter 键');
            
        }
    }

    return {
    }
}
