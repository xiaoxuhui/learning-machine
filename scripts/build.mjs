import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
export const runtimeFiles = ['index.html', 'src/model.js', 'src/visuals.js', 'src/app.js', 'src/styles.css', 'src/icon.svg'];
for (const file of runtimeFiles) {
  const target = path.join(root, 'dist', file);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.copyFile(path.join(root, file), target);
}
console.log(`Built ${runtimeFiles.length} runtime files into dist/`);
