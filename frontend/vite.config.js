import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import fs from 'fs'
import { execSync } from 'child_process'

const rootConfigPath = path.resolve(__dirname, '../config.json')
const addonConfig = JSON.parse(fs.readFileSync(rootConfigPath, 'utf-8'))
const appVersion = addonConfig.version || '0.0.0'

let gitCommit = 'unknown'
let gitBranch = 'unknown'
try {
  gitCommit = execSync('git rev-parse --short HEAD', { cwd: path.resolve(__dirname, '..') }).toString().trim()
  gitBranch = execSync('git rev-parse --abbrev-ref HEAD', { cwd: path.resolve(__dirname, '..') }).toString().trim()
} catch (_err) {
  // Leave fallbacks if git metadata is unavailable during build.
}

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [vue()],
  define: {
    __APP_VERSION__: JSON.stringify(appVersion),
    __GIT_COMMIT__: JSON.stringify(gitCommit),
    __GIT_BRANCH__: JSON.stringify(gitBranch),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
})
