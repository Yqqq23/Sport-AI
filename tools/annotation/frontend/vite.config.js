import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8001'
  // 服务器多人共用，默认 5173 一带很容易被别的同学的项目占满，Vite 会默默
  // 跳到下一个可用端口，导致本地端口转发配的端口和实际监听端口对不上。
  // 用 VITE_DEV_PORT 固定一个不容易撞车的端口，strictPort 让端口被占时
  // 直接报错退出，而不是静默换端口。
  const devPort = Number(env.VITE_DEV_PORT) || 5173

  return {
    plugins: [react()],
    server: {
      port: devPort,
      strictPort: true,
      proxy: {
        '/health': { target: proxyTarget, changeOrigin: true },
        '/basketball-classes': { target: proxyTarget, changeOrigin: true },
        '/upload-video': { target: proxyTarget, changeOrigin: true },
        '/videos': { target: proxyTarget, changeOrigin: true },
        '/jobs': { target: proxyTarget, changeOrigin: true },
        '/annotations': { target: proxyTarget, changeOrigin: true },
        '/exports': { target: proxyTarget, changeOrigin: true },
        '/static': { target: proxyTarget, changeOrigin: true },
        '/coco-exports': { target: proxyTarget, changeOrigin: true },
        '/yolo-exports': { target: proxyTarget, changeOrigin: true },
        '/dataset-versions': { target: proxyTarget, changeOrigin: true },
        '/recognition': { target: proxyTarget, changeOrigin: true },
      },
    },
  }
})
