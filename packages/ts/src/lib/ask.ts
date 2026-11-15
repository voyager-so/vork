import { execFileSync } from 'node:child_process';
import { engine, launchError } from './engine.js';

export type AskTime = number | string;
export type AskQuestion = { at?: AskTime | AskTime[] } & (
  { node: string; read: 'transform' | 'bounds'; instance?: number } |
  { node: string; read: 'screen'; instance?: number; camera?: string } |
  { field: string } |
  { buffer: string; range?: [number, number] } |
  { buffer: string; read: 'count' } |
  { image: string; read: 'info' } |
  { audio: string; read?: 'loudness' | 'spectrum'; window: [number, number]; file?: string }
);

/** Read an ordered batch of scene values without rendering. */
export function ask(scene: string, questions: readonly AskQuestion[]): unknown[] {
  try {
    const answers: unknown = JSON.parse(execFileSync(engine(), ['ask', scene, '-'], {
      input: JSON.stringify(questions), encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
      stdio: ['pipe', 'pipe', 'pipe'],
    }));
    if (!Array.isArray(answers) || answers.length !== questions.length) throw new Error('vork ask returned an invalid answer list');
    return answers;
  } catch (error) {
    const stderr = String((error as { stderr?: unknown }).stderr ?? '').trim();
    if (!stderr) launchError(error);
    throw new Error(stderr, { cause: error });
  }
}
