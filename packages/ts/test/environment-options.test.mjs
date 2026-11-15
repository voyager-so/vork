import {writeExecutable} from './executable-fixture.mjs';
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {environmentImage, sky} from '../dist/index.js';

test('environment helpers lower camelCase options to CLI JSON and reject aliases', () => {
  const dir = mkdtempSync(join(tmpdir(), 'vork-options-'));
  const binary = join(dir, 'vork');
  const previous = process.env.VORK_BIN;
  writeExecutable(binary, '#!/usr/bin/env node\nprocess.stdin.pipe(process.stdout);\n', {mode:0o755});
  process.env.VORK_BIN = binary;
  try {
    assert.deepEqual(environmentImage({file:'sky.exr', output:'weight.exr', colorSpace:'Non-Color', fps:30}),
      {file:'sky.exr', output:'weight.exr', color_space:'Non-Color', fps:30});
    assert.deepEqual(sky({sunElevation:0.5,sunRotation:1,airDensity:2,aerosolDensity:3,ozoneDensity:4,sunSize:5,sunIntensity:6,altitude:7}),
      {sun_elevation:0.5,sun_rotation:1,air_density:2,aerosol_density:3,ozone_density:4,sun_size:5,sun_intensity:6,altitude:7});
    assert.deepEqual(sky(), {});
    assert.throws(() => environmentImage({file:'x',output:'y',color_space:'sRGB'}), /unknown option/);
    for (const key of ['sun_elevation','sun_rotation','air_density','aerosol_density','ozone_density','sun_size','sun_intensity'])
      assert.throws(() => sky({[key]:1}), /unknown option/);
  } finally {
    if (previous === undefined) delete process.env.VORK_BIN; else process.env.VORK_BIN = previous;
    rmSync(dir, {recursive:true,force:true});
  }
});
