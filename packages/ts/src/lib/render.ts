import type { OutputDestination } from '../schema.js';
import { engine, launchError } from './engine.js';
import {callKeys} from './call-keys.js';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { basename, dirname, extname, join, sep } from 'node:path';
import { checkOptions } from './validate.js';

export interface RenderOptions {
  /** MP4, MOV, one still .png, PNG/control directory or %06d.exr frames; several render together. */
  out: string | OutputDestination | (string | OutputDestination)[];
  /** Still time in seconds. */
  at?: number;
  /** First frame index to render (inclusive, absolute; `--frame-start`). */
  frameStart?: number;
  /** Last frame index to render (inclusive, absolute; `--frame-end`). */
  frameEnd?: number;
  /** Render settings profile: `std/final` (default), `std/previs`, `std/draft`, `std/path` or a profile file. */
  profile?: string;
  /** Render-call fields; scene content and animation are authored in the scene. */
  set?: { resolution?: [number, number];
    fps?: number | string; seed?: number; samples?: number; denoise?: boolean };
  /** Default color; ["all"] includes controls and declared AOVs. */
  channels?: string[];
}
export interface ExposureSummary {
  frames: number;
  /** Mean alpha; exposure values are null when no frame has coverage. */
  coverage: number;
  /** Share with a display channel at 99% or more; also null for a canvas with no world, view or compositor. */
  clipped: number | null;
  near_black: number | null;
  mean: number | null;
  worst_clipped: { share: number; frame: number } | null;
  worst_near_black: { share: number; frame: number } | null;
}
export interface RenderWarning {
  code: string;
  requested: Record<string, unknown>;
  dropped: string;
}
export interface RenderResult {
  /** Visible losses and compatibility removal/rewrite notices. */
  warnings: RenderWarning[];
  /** Display-image statistics per output; EXR and control-only outputs have no entry. */
  exposure: { output: string; exposure: ExposureSummary }[];
}
/** Render a scene file, print the CLI's exposure report, and return its full-precision statistics.
 * Options are render-time choices; the scene file is unchanged.
 */
export function render(scene: string, options: RenderOptions): RenderResult {
  try {
    execFileSync(engine(), renderArguments(scene, options), { stdio: ['ignore', 'inherit', 'inherit'] });
  } catch (error) {
    launchError(error);
  }
  const exposure: RenderResult['exposure'] = [];
  const warnings = new Map<string, RenderWarning>();
  for (const destination of Array.isArray(options.out) ? options.out : [options.out]) {
    const output = typeof destination === 'string' ? destination : destination.file;
    const directory = output.endsWith('/') || output.endsWith(sep);
    const file = !directory && ['.png','.mp4','.mov','.exr'].includes(extname(output).toLowerCase());
    let name = basename(output);
    const index = name.indexOf('%06d');
    if (index >= 0) name = name.slice(0,index).replace(/[._-]+$/, '') + name.slice(index+4);
    const sidecar = name.startsWith('.') && !name.slice(1).includes('.') ? 'meta.json' : `${name}.meta.json`;
    const metadata = JSON.parse(readFileSync(file ? join(dirname(output),sidecar) : join(output,'meta.json'), 'utf8'));
    for (const warning of metadata.warnings) warnings.set(JSON.stringify(warning), warning);
    if (metadata.exposure != null) exposure.push({output, exposure: metadata.exposure});
  }
  return {exposure, warnings: [...warnings.values()]};
}
/** The `vork render` command line for a scene file and options. */
function renderArguments(scene: string, options: RenderOptions): string[] {
  checkOptions(options, ['out','at','frameStart','frameEnd','profile','set','channels']);
  const args = ['render', scene];
  for (const out of Array.isArray(options.out) ? options.out : [options.out]) args.push('--out', typeof out === 'string' ? out : JSON.stringify(out));
  if (options.at !== undefined) args.push('--at', String(options.at));
  if (options.frameStart !== undefined) args.push('--frame-start', String(options.frameStart));
  if (options.frameEnd !== undefined) args.push('--frame-end', String(options.frameEnd));
  if (options.profile !== undefined) args.push('--profile', options.profile);
  for (const [key, value] of Object.entries(options.set ?? {})) {
    checkOptions({[key]: value}, callKeys);
    args.push('--set', `${key}=${JSON.stringify(value)}`);
  }
  if (options.channels?.length) args.push('--channels', options.channels.join(','));
  return args;
}
