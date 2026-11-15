import {execFileSync} from 'node:child_process';
import {chmodSync} from 'node:fs';

// The test process never opens the executable for writing: concurrent forks
// cannot inherit its writer, even before CLOEXEC. Wait for the writer to exit;
// closing and renaming in this process would leave the same busy inode.
export function writeExecutable(path, contents, {mode = 0o700} = {}) {
  execFileSync(process.execPath, ['-e',
    'require("node:fs").writeFileSync(process.argv[1], require("node:fs").readFileSync(0))',
    path], {input: contents});
  chmodSync(path, mode);
}
