import { mkdir, open, rename, rm, realpath, lstat, readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { dirname, resolve, relative, basename } from 'node:path';
import { resourceSlots } from './resource-slots.js';
import type { Scene } from './schema.js';
import { externalBuffers, bufferObjects } from './lib/buffers.js';
export type * from './schema.js';

async function fileIdentity(path: string): Promise<string> {
  try { return await realpath(path); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    const parent = dirname(path);
    return parent === path ? path : resolve(await fileIdentity(parent), basename(path));
  }
}

/** Write external resource paths relative to the scene directory, resolving inputs against baseDir. */
async function withAssetPaths(value: Scene, baseDir: string, sceneDir: string, retained: Set<string>): Promise<Scene> {
  const assetPath = async (file: string): Promise<string> => {
    const source = resolve(baseDir, file);
    retained.add(await fileIdentity(source));
    return relative(sceneDir, resolve(await fileIdentity(dirname(source)), basename(source)));
  };
  const result = structuredClone(value);
  const files = new Map<object, Set<string>>();
  type Graph = {file?: boolean; ref?: string; branches?: readonly Graph[];
    properties?: Record<string, Graph>; items?: Graph; values?: Graph};
  const defs: Record<string, Graph> = resourceSlots.defs;
  const walk = (value: any, node: Graph, parent?: any, key?: string): void => {
    if (node.file && parent && key !== undefined && typeof value === 'string') {
      if (!files.has(parent)) files.set(parent, new Set());
      files.get(parent)!.add(key);
    }
    if (node.ref) walk(value, defs[node.ref]!);
    for (const branch of node.branches ?? []) walk(value, branch, parent, key);
    if (!value || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(node.properties ?? {})) {
      if (Object.hasOwn(value, key)) walk(value[key], child, value, key);
    }
    if (Array.isArray(value) && node.items) value.forEach((item, key) => walk(item, node.items!, value, String(key)));
    if (!Array.isArray(value) && node.values) for (const key of Object.keys(value)) walk(value[key], node.values, value, key);
  };
  walk(result, resourceSlots.root);
  for (const [parent, keys] of files) for (const key of keys) {
    const object = parent as Record<string, string>;
    object[key] = await assetPath(object[key]!);
  }
  return result;
}
// Only exact generated sibling names referenced by the replaced JSON are owned.
// Never sweep a directory: similarly named assets may belong to another scene.
async function previousSidecars(path: string): Promise<string[]> {
  let previous: unknown;
  try { previous = JSON.parse(await readFile(path, 'utf8')); }
  catch (error) {
    if (error instanceof SyntaxError || (error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
  }
  const prefix = `${basename(path)}.buffers.`;
  return [...new Set([...bufferObjects(previous)].map(([, buffer]) => buffer.file).filter((file): file is string => typeof file === 'string' && file.startsWith(prefix)
    && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.bin$/.test(file.slice(prefix.length))))]
    .map(file => resolve(dirname(path), file));
}
/** Write scene JSON with relative paths (baseDir defaults to the working directory) and a binary sidecar.
 * Serialize saves and buffer reads at one destination; copy JSON and resources for a lasting snapshot.
 */
export async function writeScene(value: Scene, path: string, baseDir = process.cwd()): Promise<string> {
  const destination = resolve(path);
  await mkdir(dirname(destination), { recursive: true });
  const destinationInfo = await lstat(destination).catch((error: NodeJS.ErrnoException) => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  });
  if (destinationInfo?.isSymbolicLink()) throw new Error('writeScene destination must not be a symbolic link');
  const sceneDir = await realpath(dirname(destination));
  const physicalDestination = resolve(sceneDir, basename(destination));
  const retained = new Set<string>();
  const scene = await withAssetPaths(value, resolve(baseDir), sceneDir, retained);
  const previous = await previousSidecars(physicalDestination);
  // Complete the sidecar before publishing JSON, then retire obsolete owned files.
  const binaryPath = `${physicalDestination}.buffers.${randomUUID()}.bin`;
  const binary = externalBuffers(scene, relative(sceneDir, binaryPath));
  const text = JSON.stringify(scene, null, 2) + '\n';
  const temporary = `${physicalDestination}.${randomUUID()}.tmp`;
  let temporaryCreated = false;
  let binaryCreated = false;
  try {
    if (binary) {
      const file = await open(binaryPath, 'wx', 0o666);
      binaryCreated = true;
      try { await file.writeFile(binary); } finally { await file.close(); }
    }
    const file = await open(temporary, 'wx', 0o666);
    temporaryCreated = true;
    try {
      await file.writeFile(text);
      if (destinationInfo?.isFile()) await file.chmod(destinationInfo.mode & 0o777);
    } finally { await file.close(); }
    await rename(temporary, physicalDestination);
  } catch (error) {
    if (temporaryCreated) await rm(temporary, {force:true});
    if (binaryCreated) await rm(binaryPath, {force:true});
    throw error;
  }
  for (const old of previous) {
    if (!retained.has(await fileIdentity(old))) {
      const info = await lstat(old).catch((error: NodeJS.ErrnoException) => {
        if (error.code !== 'ENOENT') throw error;
        return undefined;
      });
      if (info?.isFile()) await rm(old, {force:true});
    }
  }
  return destination;
}

export { box, sphere, grid, cylinder, cone, torus, capsule, revolve, roundedBox } from './lib/shapes.js';
export type { GeometryFragment } from './lib/shapes.js';
export { text, type TextOptions } from './lib/text.js';
export { extrude, type ExtrudeOptions } from './lib/extrude.js';
export { sweep, type SweepOptions } from './lib/sweep.js';
export { render, type RenderOptions, type RenderResult, type RenderWarning, type ExposureSummary } from './lib/render.js';
export { ask, type AskQuestion, type AskTime } from './lib/ask.js';
export { Session, type SessionOptions, type SessionOpenOptions, type SessionRenderOptions, type SessionOpenResult, type SessionRenderResult, type SessionReuse, type ChangedPath } from './lib/session.js';
export { environmentImage, sky, type EnvironmentImageOptions, type EnvironmentFragment } from './lib/environment.js';

export {exportAsset} from './lib/export.js';
export type {ExportOptions, ExportReport, ExportIssue} from './lib/export.js';
