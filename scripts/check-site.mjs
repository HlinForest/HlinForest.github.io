import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';

const root = resolve('dist');
const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const file = join(dir, name);
    statSync(file).isDirectory() ? walk(file) : files.push(file);
  }
};
walk(root);

const htmlFiles = files.filter((file) => extname(file) === '.html');
const errors = [];
const resolveTarget = (from, raw) => {
  const clean = decodeURI(raw.split('#')[0].split('?')[0]);
  if (!clean) return from;
  const path = clean.startsWith('/') ? join(root, clean) : resolve(dirname(from), clean);
  if (!normalize(path).startsWith(root + sep) && normalize(path) !== root) return null;
  if (existsSync(path) && statSync(path).isFile()) return path;
  if (existsSync(path) && statSync(path).isDirectory() && existsSync(join(path, 'index.html'))) return join(path, 'index.html');
  if (!extname(path) && existsSync(join(path, 'index.html'))) return join(path, 'index.html');
  return path;
};

for (const file of htmlFiles) {
  const html = readFileSync(file, 'utf8');
  const refs = [...html.matchAll(/(?:href|src)=["']([^"']+)["']/g)].map((match) => match[1]);
  for (const ref of refs) {
    if (ref.includes('${') || /^(?:https?:|mailto:|tel:|data:|javascript:|#)/i.test(ref)) continue;
    const target = resolveTarget(file, ref);
    if (target && !existsSync(target)) errors.push(`${file.slice(root.length)} -> ${ref}`);
  }
}

if (errors.length) {
  console.error(`Broken internal references (${errors.length}):\n${errors.slice(0, 80).join('\n')}`);
  process.exit(1);
}

console.log(`Checked ${htmlFiles.length} HTML pages and ${files.length} generated files: no broken internal references.`);
