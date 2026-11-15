import { spawn, type ChildProcess } from 'node:child_process';
import { createInterface } from 'node:readline';
import { engine, launchError } from './engine.js';
import { callKeys } from './call-keys.js';
import { checkOptions } from './validate.js';
import type { RenderOptions, RenderResult } from './render.js';
import type { AskQuestion } from './ask.js';

/** What an open kept: nothing dropped (identical document), values written into the kept installation,
 * the installation dropped for the next render to install on the kept device, or dropped with a program to compile. */
export type SessionReuse = 'reused' | 'updated' | 'reinstalled' | 'recompiled';
/** The reply field carrying the reuse class; named once so a rename is one edit. */
const REUSE_FIELD = 'reuse';
/** An authored path that differed, in the animation-track grammar; a number is an unnamed list position. */
export type ChangedPath = (string | number)[];
export type SessionOpenResult = { revision: number; changed: ChangedPath[] } & Record<typeof REUSE_FIELD, SessionReuse>;
export type SessionRenderResult = RenderResult & {
  /** The document revision that rendered. */
  revision: number;
  /** Installations this session has performed. */
  installs: number;
  /** True for an early picture that is not the final one; never delivery. */
  provisional: boolean;
} & Record<typeof REUSE_FIELD, SessionReuse>;
export interface SessionRenderOptions extends RenderOptions {
  /** The first picture available instead of the final one; the result says `provisional: true`. */
  provisional?: boolean;
}
export interface SessionOpenOptions {
  /** Directory the document's relative files resolve against (object form); default the current directory. */
  base?: string;
}
export interface SessionOptions {
  /** Hardware GPU index in PCI bus order, or "cpu"; default the high-performance GPU. */
  gpu?: number | 'cpu';
}
/** A kept-alive `vork session` worker: the GPU, compiled pipelines and the installed scene
 * survive between renders. open a scene, render, close. To edit, open again with the whole
 * edited scene; the worker compares it with the open one and reuses what it can. Requests run
 * in order, each with the values its arguments had when it was called; a worker that exits
 * mid-request fails that request.
 */
export class Session {
  private child: ChildProcess;
  private replies: AsyncIterator<string>;
  private queue: Promise<unknown> = Promise.resolve();
  private next = 1;
  private closed = false;
  constructor(options: SessionOptions = {}) {
    checkOptions(options, ['gpu']);
    const args = ['session', ...(options.gpu !== undefined ? ['--gpu', String(options.gpu)] : [])];
    this.child = spawn(engine(), args, { stdio: ['pipe', 'pipe', 'inherit'] });
    this.child.on('error', () => {});
    this.replies = createInterface({ input: this.child.stdout!, crlfDelay: Infinity })[Symbol.asyncIterator]();
  }
  /** Install the whole scene, a file path or a scene object; omitted fields are defaults. Opening again
   * replaces the document; a failed open keeps the previous revision. */
  open(scene: string | object, options: SessionOpenOptions = {}): Promise<SessionOpenResult> {
    checkOptions(options, ['base']);
    const body: Record<string, unknown> = typeof scene === 'string' ? { op: 'open', scene } : { op: 'open', document: scene };
    if (options.base !== undefined) body.base = options.base;
    return this.request(body).then(reply => ({
      revision: reply.revision as number, [REUSE_FIELD]: reply[REUSE_FIELD] as SessionReuse, changed: reply.changed as ChangedPath[],
    }));
  }
  /** Read an ordered batch from the open scene without rendering. */
  ask(questions: readonly AskQuestion[]): Promise<unknown[]> {
    const asked = questions.length;
    return this.request({ op: 'ask', questions }).then(reply => {
      if (!Array.isArray(reply.answers) || reply.answers.length !== asked) throw new Error('vork ask returned an invalid answer list');
      return reply.answers;
    });
  }
  /** Render the open document with today's render options. `provisional: true` returns the first
   * picture available (one sample, no denoise), labelled; render again for the final. */
  render(options: SessionRenderOptions): Promise<SessionRenderResult> {
    checkOptions(options, ['out', 'at', 'frameStart', 'frameEnd', 'profile', 'set', 'channels', 'provisional']);
    for (const key of Object.keys(options.set ?? {})) checkOptions({ [key]: true }, callKeys);
    const body: Record<string, unknown> = { op: 'render', out: Array.isArray(options.out) ? options.out : [options.out] };
    if (options.at !== undefined) body.at = options.at;
    if (options.frameStart !== undefined) body.frame_start = options.frameStart;
    if (options.frameEnd !== undefined) body.frame_end = options.frameEnd;
    if (options.profile !== undefined) body.profile = options.profile;
    if (options.set !== undefined) body.set = options.set;
    if (options.channels?.length) body.channels = options.channels;
    if (options.provisional) body.provisional = true;
    return this.request(body).then(reply => {
      const result = reply.result as { warnings: RenderResult['warnings']; exposure: RenderResult['exposure'] };
      return { warnings: result.warnings, exposure: result.exposure, revision: reply.revision as number,
        [REUSE_FIELD]: reply[REUSE_FIELD] as SessionReuse, installs: reply.installs as number, provisional: reply.provisional === true };
    });
  }
  /** Release the document and end the worker. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.request({ op: 'close' }, true);
    this.child.stdin!.end();
  }
  private request(body: Record<string, unknown>, closing = false): Promise<Record<string, unknown>> {
    if (this.closed && !closing) return Promise.reject(new Error('vork session is closed'));
    // The request is its wire bytes from this call on: the queue holds no reference to the
    // caller's scene or options, so changing them after the call changes nothing sent.
    const op = body.op;
    const line = JSON.stringify({ id: this.next, ...body }) + '\n';
    const id = this.next++;
    const run = async (): Promise<Record<string, unknown>> => {
      // Request-local handlers, removed when the request settles so a long
      // session does not accumulate listeners on the child.
      let onError!: (error: Error) => void;
      let onExit!: (code: number | null) => void;
      const exit = new Promise<never>((_, reject) => {
        onError = error => { try { launchError(error); } catch (launch) { reject(launch); } };
        onExit = code => reject(new Error(`vork session ended (exit ${code}) before replying to ${op}`));
        this.child.once('error', onError);
        this.child.once('exit', onExit);
      });
      exit.catch(() => {});
      try {
        const written = new Promise<void>((resolve, reject) => {
          this.child.stdin!.write(line, error => error ? reject(error) : resolve());
        });
        await Promise.race([written, exit]);
        const answer = await Promise.race([this.replies.next(), exit]);
        if (answer.done) throw new Error(`vork session ended before replying to ${op}`);
        const reply = JSON.parse(answer.value) as Record<string, unknown>;
        if (reply.id !== id) throw new Error(`vork session replied out of order (expected ${id}, got ${reply.id})`);
        if (reply.ok !== true) throw new Error(`${reply.kind}: ${reply.error}`);
        return reply;
      } finally {
        this.child.off('error', onError);
        this.child.off('exit', onExit);
      }
    };
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => {});
    return result;
  }
}
