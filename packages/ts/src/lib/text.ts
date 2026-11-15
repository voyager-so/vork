import { stdCommand } from './cli.js';
import { checkName, checkOptions } from './validate.js';
import type { GeometryFragment } from './shapes.js';

export interface TextOptions {
  body: string;
  /** TrueType/OpenType font file; the font's licence remains the author's responsibility. */
  font: string;
  size?: number;
  /** Half the extrusion thickness, as in Blender Text. */
  extrude?: number;
  bevelDepth?: number;
  bevelResolution?: number;
  alignX?: 'left' | 'center' | 'right';
  alignY?: 'top_baseline' | 'top' | 'center' | 'bottom_baseline' | 'bottom';
}
/** Font file → ordinary geometry, in XY with +Z front. Uses the installed vork std loader
 * (override its executable with VORK_BIN). No text object enters the scene.
 * @help font:TrueType/OpenType file; extrude:half-depth; bevel_depth:metres
 */
export function text(name: string = "text", options: TextOptions): GeometryFragment {
  checkName(name, 'text name');
  checkOptions(options, ['body','font','size','extrude','bevelDepth','bevelResolution','alignX','alignY']);
  const {bevelDepth, bevelResolution, alignX, alignY, ...rest} = options;
  // vork std text takes the scene's snake_case keys.
  return stdCommand('text', {name, ...rest, bevel_depth: bevelDepth, bevel_resolution: bevelResolution, align_x: alignX, align_y: alignY}) as GeometryFragment;
}
