import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import fs from 'fs'
import { execSync } from 'child_process'

const rootConfigPath = path.resolve(__dirname, '../config.json')
const frontendPkgPath = path.resolve(__dirname, './package.json')

let appVersion = '0.0.0'
try {
  if (fs.existsSync(rootConfigPath)) {
    const addonConfig = JSON.parse(fs.readFileSync(rootConfigPath, 'utf-8'))
    appVersion = addonConfig.version || appVersion
  } else if (fs.existsSync(frontendPkgPath)) {
    const frontendPkg = JSON.parse(fs.readFileSync(frontendPkgPath, 'utf-8'))
    appVersion = frontendPkg.version || appVersion
  }
} catch (_err) {
  // Keep default if version metadata can't be read.
}

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
