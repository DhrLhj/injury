import { onMounted, onBeforeUnmount } from 'vue';

export default function useRecordVideo() {
    // ... 其他代码

    onMounted(() => {
        setTimeout(() => {
            document.addEventListener('keydown', handleKeyDown);
        }, 10);
    });

    onBeforeUnmount(() => {
        document.removeEventListener('keydown', handleKeyDown);
    });

    function handleKeyDown(event) {
        if (event.key === '9') {
            console.log('按下了 9 键');

        }
    }

    return {
    }
}
