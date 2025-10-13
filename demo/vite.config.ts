import { defineConfig } from 'vite';
import { copyFileSync, mkdirSync, existsSync } from 'fs';
import { resolve } from 'path';

export default defineConfig({
  plugins: [
    {
      name: 'copy-assets',
      buildStart() {
        // Copy emoji images
        const emojiSrcDir = resolve(__dirname, '../data/emojis');
        const emojiDestDir = resolve(__dirname, 'public/emojis');

        if (!existsSync(emojiDestDir)) {
          mkdirSync(emojiDestDir, { recursive: true });
        }

        for (let i = 0; i < 10; i++) {
          const srcFile = resolve(emojiSrcDir, `emoji_${i}.png`);
          const destFile = resolve(emojiDestDir, `emoji_${i}.png`);
          if (existsSync(srcFile)) {
            copyFileSync(srcFile, destFile);
          }
        }

        // Copy ONNX models
        const modelsSrcDir = resolve(__dirname, '../models');
        const modelsDestDir = resolve(__dirname, 'public/models');

        if (!existsSync(modelsDestDir)) {
          mkdirSync(modelsDestDir, { recursive: true });
        }

        for (let i = 0; i < 10; i++) {
          const srcFile = resolve(modelsSrcDir, `emoji_${i}.onnx`);
          const destFile = resolve(modelsDestDir, `emoji_${i}.onnx`);
          if (existsSync(srcFile)) {
            copyFileSync(srcFile, destFile);
          }
        }
      },
    },
  ],
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {
      allow: ['..'],
    },
  },
  resolve: {
    alias: {
      'onnxruntime-web': resolve(
        __dirname,
        'node_modules/onnxruntime-web/dist/ort.min.mjs'
      ),
    },
  },
});
