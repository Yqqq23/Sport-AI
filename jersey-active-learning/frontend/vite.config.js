import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const proxyTarget = env.VITE_PROXY_TARGET || 'http://127.0.0.1:8001'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/health': { target: proxyTarget, changeOrigin: true },
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
