#!/usr/bin/env python3
"""Exercise the real POSIX and PowerShell installers against local release fixtures."""
import hashlib
import io
import os
import platform
from pathlib import Path
import subprocess
import tarfile
import tempfile
import shutil
import unittest
import zipfile

from test_support import write_executable

INSTALLER = Path(__file__).resolve().parents[1] / 'install.sh'
TAR_READS_XZ = os.name != 'nt' and 'liblzma' in subprocess.run(['tar', '--version'], capture_output=True, text=True).stdout
PS_INSTALLER = INSTALLER.with_name('install.ps1')

@unittest.skipIf(os.name == 'nt', 'POSIX installer')
class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, PREFIX=str(self.root / 'prefix'), HOME=str(self.root / 'home'),
                        VERSION='test', VORK_DOWNLOAD_BASE=self.root.as_uri())
        self.archive = self.root / ('vork-test-darwin-arm64.tar.gz' if platform.system() == 'Darwin' else 'vork-test-linux-x64.tar.gz')
        self.fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, member='vork', binary=b'#!/bin/sh\necho fixture "$@"\n'):
        with tarfile.open(self.archive, 'w:gz') as tar:
            for name, data, mode in [(member, binary, 0o755),
                                     ('examples/scene.json', b'{}', 0o644), # export-example-generated: synthetic tar member
                                     ('slang/bin/slangc', b'compiler', 0o755),
                                     ('oidn/lib/model', b'model', 0o644),
                                     ('NOTICE', b'notices', 0o644)]:
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = mode
                tar.addfile(info, io.BytesIO(data))
        self.checksum()

    def checksum(self):
        (self.root / 'SHA256SUMS').write_text(hashlib.sha256(self.archive.read_bytes()).hexdigest() +
                                            '  ' + self.archive.name + '\n')

    def run_installer(self, *args):
        return subprocess.run(['sh', str(INSTALLER), *args], cwd=self.root, env=self.env,
                              capture_output=True, text=True)

    def test_global_install_and_setup(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'prefix/lib/vork/NOTICE').exists())
        self.assertIn('fixture setup', result.stdout)
        self.assertTrue((self.root / 'prefix/bin/vork').is_symlink())
        self.assertTrue((self.root / 'prefix/lib/vork/slang/bin/slangc').exists())
        self.assertTrue((self.root / 'prefix/lib/vork/oidn/lib/model').exists())

    def bundled_installer(self):
        installer = self.root / 'install.sh'
        shutil.copyfile(INSTALLER, installer)
        self.env['VORK_DOWNLOAD_BASE'] = 'file:///no-downloads-allowed'
        return subprocess.run(['sh', str(installer)], cwd='/', env=self.env,
                              capture_output=True, text=True)

    def test_bundled_archive_from_another_directory(self):
        result = self.bundled_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Using bundled archive:', result.stdout)
        self.assertIn('fixture setup', result.stdout)

    def test_bundled_archive_requires_checksum(self):
        (self.root / 'SHA256SUMS').unlink()
        result = self.bundled_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires SHA256SUMS', result.stderr)
        self.assertFalse((self.root / 'prefix').exists())

    def test_bundled_corruption_fails(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'corruption')
        self.assertNotEqual(self.bundled_installer().returncode, 0)
        self.assertFalse((self.root / 'prefix').exists())

    def test_bundled_upgrade_replaces_tree_and_preserves_it_on_corruption(self):
        self.assertEqual(self.bundled_installer().returncode, 0)
        installed = self.root / 'prefix/lib/vork'
        stale = installed / 'dropped-by-new-release'
        stale.write_text('old')
        result = self.bundled_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        notice = (installed / 'NOTICE').read_bytes()
        self.archive.write_bytes(self.archive.read_bytes() + b'corruption')
        result = self.bundled_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Nothing was installed', result.stderr)
        self.assertEqual((installed / 'NOTICE').read_bytes(), notice)
        self.assertTrue((self.root / 'prefix/bin/vork').resolve().is_file())

    def test_skip_setup(self):
        self.env['VORK_SKIP_SETUP'] = '1'
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('fixture setup', result.stdout)
        self.assertIn('fixture --version', result.stdout)

    def test_relative_prefix(self):
        self.env['PREFIX'] = 'relative-prefix'
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'relative-prefix/bin/vork').resolve().is_file())

    def test_corruption_fails_before_install(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'corruption')
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.root / 'prefix/bin/vork').exists())

    def test_missing_checksum_fails(self):
        (self.root / 'SHA256SUMS').write_text('')
        self.assertNotEqual(self.run_installer().returncode, 0)

    def test_duplicate_checksum_fails(self):
        sums = self.root / 'SHA256SUMS'; sums.write_text(sums.read_text() * 2)
        self.assertNotEqual(self.run_installer().returncode, 0)

    def test_traversal_fails(self):
        self.fixture('../vork')
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.root / 'prefix/bin/vork').exists())

    def test_rerun_removes_files_a_release_dropped(self):
        self.assertEqual(self.run_installer().returncode, 0)
        stale = self.root / 'prefix/lib/vork/dropped-by-new-release'
        stale.write_text('old')
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(stale.exists())
        self.assertTrue((self.root / 'prefix/lib/vork/NOTICE').exists())
        self.assertEqual([p.name for p in (self.root / 'prefix/lib').iterdir()], ['vork'])

    def test_failed_download_names_the_url_and_keeps_the_install(self):
        self.assertEqual(self.run_installer().returncode, 0)
        self.env['VORK_DOWNLOAD_BASE'] = (self.root / 'absent').as_uri()
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not download', result.stderr)
        self.assertIn('absent/SHA256SUMS', result.stderr)
        self.assertTrue((self.root / 'prefix/lib/vork/NOTICE').exists())

    def test_default_prefix_adds_path_once_for_a_new_shell(self):
        home = self.root / 'home'; home.mkdir()
        del self.env['PREFIX']
        self.env.update(SHELL='/bin/zsh', PATH='/usr/bin:/bin:/usr/sbin:/sbin')
        for _ in range(2):
            result = self.run_installer()
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((home / '.zshenv').read_text().count('export PATH="$HOME/.local/bin:$PATH"'), 1)
        self.assertIn('New shells that read', result.stdout)
        self.assertIn(str(home / '.local/bin/vork'), result.stdout)

    def test_custom_prefix_path_opt_in_and_no_modify_precedence(self):
        home = self.root / 'home'; home.mkdir()
        self.env.update(SHELL='/bin/zsh', PREFIX=str(self.root / "Zoë O'Neil $(touch injected) `touch injected2`"))
        rc = home / '.zshenv'
        for opt_in, opt_out, modifies in [('0', '0', False), ('1', '0', True), ('1', '1', False), ('0', '1', False)]:
            with self.subTest(opt_in=opt_in, opt_out=opt_out):
                if rc.exists(): rc.unlink()
                self.env.update(VORK_MODIFY_PATH=opt_in, VORK_NO_MODIFY_PATH=opt_out)
                for _ in range(2):
                    result = self.run_installer()
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(rc.exists(), modifies)
                if modifies:
                    self.assertEqual(rc.read_text().count('# Added by the vork installer'), 1)
                    result = subprocess.run(['sh', '-c', '. "$HOME/.zshenv"; command -v vork'],
                                            env=self.env, cwd=self.root, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), self.env['PREFIX'] + '/bin/vork')
                self.assertFalse((self.root / 'injected').exists())
                self.assertFalse((self.root / 'injected2').exists())

    @unittest.skipUnless(shutil.which('bash'), 'needs bash')
    def test_bash_login_shell_finds_vork_past_a_stock_early_return(self):
        home = self.root / 'home'; home.mkdir()
        (home / '.bashrc').write_text('[ -z "$PS1" ] && return\nalias ll="ls -l"\n')
        (home / '.profile').write_text('if [ -f ~/.bashrc ]; then . ~/.bashrc; fi\nmesg n 2>/dev/null || true\n')
        del self.env['PREFIX']
        self.env.update(SHELL='/bin/bash', PATH='/usr/bin:/bin:/usr/sbin:/sbin')
        for _ in range(2):
            self.assertEqual(self.run_installer().returncode, 0)
        for name in ('.bashrc', '.profile'):
            self.assertEqual((home / name).read_text().count('export PATH="$HOME/.local/bin:$PATH"'), 1)
        env = {'HOME': str(home), 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'}
        login = subprocess.run(['bash', '-lc', 'vork --version'], env=env, capture_output=True, text=True)
        self.assertEqual(login.returncode, 0, login.stderr)
        self.assertIn('fixture --version', login.stdout)

    def test_bash_keeps_an_existing_bash_profile_as_the_login_file(self):
        home = self.root / 'home'; home.mkdir()
        (home / '.bash_profile').write_text('# mine\n')
        del self.env['PREFIX']
        self.env.update(SHELL='/bin/bash', PATH='/usr/bin:/bin')
        self.assertEqual(self.run_installer().returncode, 0)
        self.assertIn('.local/bin', (home / '.bash_profile').read_text())
        self.assertFalse((home / '.profile').exists())

    @unittest.skipUnless(shutil.which('bash'), 'needs bash')
    def test_bundled_default_install_is_found_by_a_login_shell(self):
        home = self.root / 'home'; home.mkdir()
        (home / '.bashrc').write_text('[ -z "$PS1" ] && return\n')
        (home / '.profile').write_text('if [ -f ~/.bashrc ]; then . ~/.bashrc; fi\n')
        del self.env['PREFIX']
        self.env.update(SHELL='/bin/bash', PATH='/usr/bin:/bin:/usr/sbin:/sbin')
        result = self.bundled_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Using bundled archive:', result.stdout)
        login = subprocess.run(['bash', '-lc', 'vork --version'], cwd='/',
                               env={'HOME': str(home), 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'},
                               capture_output=True, text=True)
        self.assertEqual(login.returncode, 0, login.stderr)
        self.assertIn('fixture --version', login.stdout)

    def test_custom_prefix_and_opt_out_leave_startup_files_alone(self):
        home = self.root / 'home'; home.mkdir()
        self.env.update(SHELL='/bin/zsh', PATH='/usr/bin:/bin:/usr/sbin:/sbin')
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prefix/bin':\"$PATH\"", result.stdout)
        del self.env['PREFIX']
        self.env['VORK_NO_MODIFY_PATH'] = '1'
        self.assertEqual(self.run_installer().returncode, 0)
        self.assertFalse((home / '.zshenv').exists())

    def compact_fixture(self):
        """A .tar.xz beside the .tar.gz, both in SHA256SUMS; its binary says which archive was installed."""
        compact = self.archive.with_name(self.archive.name.removesuffix('.gz') + '.xz')
        data = b'#!/bin/sh\necho fixture-xz "$@"\n'
        with tarfile.open(compact, 'w:xz') as tar:
            info = tarfile.TarInfo('vork'); info.size = len(data); info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        (self.root / 'SHA256SUMS').write_text(''.join(
            hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n' for path in (self.archive, compact)))
        return compact

    def path_without_xz(self, tar_reads_xz):
        """PATH with every command but xz. `tar_reads_xz` False also makes tar report no xz support of its own."""
        tools = self.root / 'tools'; tools.mkdir()
        for folder in os.environ['PATH'].split(os.pathsep):
            for entry in (sorted(Path(folder).iterdir()) if Path(folder).is_dir() else []):
                if entry.name != 'xz' and not (tools / entry.name).exists() and not (tools / entry.name).is_symlink():
                    (tools / entry.name).symlink_to(entry)
        if not tar_reads_xz:
            (tools / 'tar').unlink()
            write_executable(tools / 'tar', '#!/bin/sh\nif [ "$1" = --version ]; then echo "tar (fixture)"; exit 0; fi\nexec "' + shutil.which('tar') + '" "$@"\n')
        self.env['PATH'] = str(tools)

    @unittest.skipUnless(shutil.which('xz'), 'needs the xz program')
    def test_xz_program_installs_the_tar_xz(self):
        self.compact_fixture()
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixture-xz setup', result.stdout)
        dry = self.run_installer('--dry-run')
        self.assertIn(self.archive.name.removesuffix('.gz') + '.xz', dry.stdout)

    @unittest.skipUnless(TAR_READS_XZ, 'needs a tar with its own xz support (bsdtar with liblzma)')
    def test_tar_with_xz_support_installs_the_tar_xz_without_the_xz_program(self):
        self.compact_fixture()
        self.path_without_xz(tar_reads_xz=True)
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixture-xz setup', result.stdout)

    def test_without_xz_the_tar_gz_is_installed(self):
        self.compact_fixture()
        self.path_without_xz(tar_reads_xz=False)
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixture setup', result.stdout)
        self.assertNotIn('fixture-xz', result.stdout)
        dry = self.run_installer('--dry-run')
        self.assertIn(self.archive.name, dry.stdout)
        self.assertNotIn('.tar.xz', dry.stdout)

    @unittest.skipUnless(shutil.which('xz') or TAR_READS_XZ, 'needs xz support')
    def test_tar_xz_checksum_is_verified(self):
        compact = self.compact_fixture()
        compact.write_bytes(compact.read_bytes() + b'corruption')
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Checksum mismatch for ' + compact.name, result.stderr)
        self.assertFalse((self.root / 'prefix/bin/vork').exists())

    @unittest.skipUnless(shutil.which('xz') or TAR_READS_XZ, 'needs xz support')
    def test_bundled_tar_xz_is_preferred(self):
        self.compact_fixture()
        result = self.bundled_installer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Using bundled archive:', result.stdout)
        self.assertIn('fixture-xz setup', result.stdout)

    def test_macos_floor_is_checked_before_download(self):
        tools = self.root / 'tools'; tools.mkdir()
        for name, content in [('uname', '#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n'),
                              ('sw_vers', '#!/bin/sh\necho "$TEST_MACOS_VERSION"\n')]:
            path = tools / name; write_executable(path, content)
        self.env['PATH'] = str(tools) + os.pathsep + os.environ['PATH']
        # The installer is a standalone download, so it spells the floor itself; dist/macos-minimum.txt owns it.
        major, minor = map(int, INSTALLER.with_name('dist').joinpath('macos-minimum.txt').read_text().strip().split('.'))
        self.assertEqual(minor, 0, 'install.sh compares the major version only')
        for version in [f'{major - 1}.7', '11.0', 'unknown', '']:
            self.env['TEST_MACOS_VERSION'] = version
            result = self.run_installer('--dry-run')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f'macOS {major}+', result.stderr)
            self.assertIn('Nothing was downloaded', result.stderr)
        for version in [f'{major}.0', f'{major}.7', '26.0', '26.1', '27.0']:
            self.env['TEST_MACOS_VERSION'] = version
            self.assertEqual(self.run_installer('--dry-run').returncode, 0)
        self.assertFalse((self.root / 'prefix').exists())

    def test_extraction_failure_is_bounded_and_preserves_old_tree(self):
        self.assertEqual(self.run_installer().returncode, 0)
        notice = (self.root / 'prefix/lib/vork/NOTICE').read_bytes()
        tools = self.root / 'tools'; tools.mkdir()
        real_tar = shutil.which('tar')
        fake = tools / 'tar'
        write_executable(fake, '#!/bin/sh\nif [ "$1" = -xzf ]; then i=0; while [ "$i" -lt 2000 ]; do echo "tar: No space left on device" >&2; i=$((i+1)); done; exit 2; fi\nexec "' + real_tar + '" "$@"\n')
        self.env['PATH'] = str(tools) + os.pathsep + os.environ['PATH']
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not extract', result.stderr)
        self.assertIn('No space left on device', result.stderr)
        self.assertEqual(result.stderr.count('tar: No space left on device'), 8)
        self.assertEqual((self.root / 'prefix/lib/vork/NOTICE').read_bytes(), notice)
        self.assertEqual([p.name for p in (self.root / 'prefix/lib').iterdir()], ['vork'])

    def test_failed_publication_restores_previous_tree(self):
        self.assertEqual(self.run_installer().returncode, 0)
        tools = self.root / 'tools'; tools.mkdir()
        real_mv = shutil.which('mv')
        fake = tools / 'mv'
        write_executable(fake, '#!/bin/sh\ncase "$1" in */.vork-new.*) echo "injected rename failure" >&2; exit 1;; esac\nexec "' + real_mv + '" "$@"\n')
        self.env['PATH'] = str(tools) + os.pathsep + os.environ['PATH']
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / 'prefix/bin/vork').resolve().is_file())
        self.assertEqual([p.name for p in (self.root / 'prefix/lib').iterdir()], ['vork'])

    def test_printed_commands_are_shell_literals(self):
        self.env['PREFIX'] = str(self.root / "Zoë O'Neil $(touch injected) `touch injected2`")
        self.fixture(binary=b'#!/bin/sh\n[ "$1" != setup ] || [ "${FIXTURE_FAIL_SETUP:-}" != 1 ]\n')
        self.env['FIXTURE_FAIL_SETUP'] = '1'
        result = self.run_installer()
        self.assertNotEqual(result.returncode, 0)
        retry = result.stderr.split('then run: ', 1)[1].splitlines()[0]
        env = {**self.env, 'FIXTURE_FAIL_SETUP': '0'}
        pasted = subprocess.run(['sh', '-c', retry], cwd=self.root, env=env, capture_output=True)
        self.assertEqual(pasted.returncode, 0, pasted.stderr)
        path_line = next(line.strip() for line in result.stderr.splitlines() if line.strip().startswith('export PATH='))
        pasted = subprocess.run(['sh', '-c', path_line + '; command -v vork'], cwd=self.root,
                                env=env, capture_output=True, text=True)
        self.assertEqual(pasted.stdout.strip(), self.env['PREFIX'] + '/bin/vork')
        self.assertFalse((self.root / 'injected').exists())
        self.assertFalse((self.root / 'injected2').exists())

    def test_dry_run_downloads_and_writes_nothing(self):
        self.env['VORK_DOWNLOAD_BASE'] = 'https://github.com/voyager-so/vork/releases/download/vtest'
        result = self.run_installer('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SHA256SUMS', result.stdout)
        self.assertFalse((self.root / 'prefix').exists())

@unittest.skipUnless(shutil.which('pwsh') or (os.name == 'nt' and shutil.which('powershell')), 'needs PowerShell')
class PowerShellLiteralTests(unittest.TestCase):
    def test_custom_prefix_path_opt_in_and_no_modify_precedence(self):
        source = PS_INSTALLER.read_text()
        block = source[source.index('    # Let a new window'):source.index("    if ($env:VORK_SKIP_SETUP")]
        quote = source[source.index('function Quote-ShellLiteral'):source.index('$bin = $prefix')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = str(root / "Zoë O'Neil $(throw 'injected') `x`")
            script = root / 'test.ps1'
            script.write_text("$ErrorActionPreference = 'Stop'\n" +
                "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n" + quote +
                "\n$bin = @'\n" + folder + "\n'@\n$quotedBin = Quote-ShellLiteral $bin\n" +
                "$global:added = @(); function Add-UserPath([string]$folder) { $global:added += $folder }\n" +
                "$env:Path = 'existing'; $env:PREFIX = if ($env:TEST_CUSTOM -eq '1') { $bin } else { '' }\n" +
                block + block +
                "\n@{added=$global:added; path=$env:Path} | ConvertTo-Json -Compress\n", encoding='utf-8-sig')
            shell = shutil.which('pwsh') or 'powershell'
            import json
            for custom in ['0', '1']:
                for opt_in in ['0', '1']:
                    for opt_out in ['0', '1']:
                        with self.subTest(custom=custom, opt_in=opt_in, opt_out=opt_out):
                            result = subprocess.run([shell, '-NoProfile', '-File', str(script)], capture_output=True, text=True, encoding='utf-8',
                                env={**os.environ, 'TEST_CUSTOM':custom, 'VORK_MODIFY_PATH':opt_in, 'VORK_NO_MODIFY_PATH':opt_out})
                            self.assertEqual(result.returncode, 0, result.stderr)
                            data = json.loads(result.stdout)
                            modifies = opt_out != '1' and (custom != '1' or opt_in == '1')
                            self.assertEqual(data['added'], [folder, folder] if modifies else [])
                            self.assertEqual(data['path'], folder + ';existing' if modifies else 'existing')

    def test_prefix_form_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            location = root / 'PowerShell location'
            location.mkdir()
            script = root / 'prefix-matrix.ps1'
            script.write_text(
                "$ErrorActionPreference = 'Stop'\n"
                "[Environment]::CurrentDirectory = $env:TEST_PROCESS_DIRECTORY\n"
                "Set-Location -LiteralPath $env:TEST_LOCATION\n"
                "if ((Get-Location).ProviderPath -eq [Environment]::CurrentDirectory) { throw 'Test needs distinct directories' }\n"
                "& $env:TEST_INSTALLER -DryRun\n", encoding='utf-8-sig')
            shell = shutil.which('pwsh') or 'powershell'
            cases = [('vork', str(location / 'vork')),
                     ('./missing/../vork/', str(location / 'vork')),
                     (str(root / 'absolute') + os.sep, str(root / 'absolute')),
                     (root.anchor, None),
                     ('\\', None), ('/', None), ('\\\\', None)]
            if os.name == 'nt':
                cases += [('C:\\vork', 'C:\\vork'),
                          ('.\\vork', str(location / 'vork')),
                          ('C:vork', str(location / 'vork')),
                          ('\\vork', str(Path(root.anchor) / 'vork')),
                          ('/vork', str(Path(root.anchor) / 'vork')),
                          ('C:\\', None), ('C:/', None),
                          ('C:\\missing\\..\\', None),
                          ('\\missing\\..', None),
                          ('C:\\vork\\', 'C:\\vork'),
                          ('\\\\server\\share\\vork', '\\\\server\\share\\vork'),
                          ('\\\\server\\share\\vork\\', '\\\\server\\share\\vork'),
                          ('\\\\server\\share\\', None),
                          ('./missing/../', str(location))]
            for prefix, expected in cases:
                with self.subTest(prefix=prefix):
                    result = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)],
                        cwd=root, env={**os.environ, 'PREFIX': prefix,
                            'TEST_PROCESS_DIRECTORY': str(root), 'TEST_LOCATION': str(location),
                            'TEST_INSTALLER': str(PS_INSTALLER), 'VERSION': 'test',
                            'VORK_DOWNLOAD_BASE': 'file:///no-downloads-allowed',
                            'PROCESSOR_ARCHITECTURE': 'AMD64', 'PROCESSOR_ARCHITEW6432': 'AMD64'},
                        capture_output=True, text=True, timeout=15)
                    if expected is None:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('PREFIX must name an install directory', result.stderr)
                        self.assertNotIn('Archive:', result.stdout)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn('Destination: ' + expected, result.stdout.splitlines())
                        self.assertIn('Binary: ' + expected + '\\vork.exe', result.stdout.splitlines())
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ['PowerShell location', 'prefix-matrix.ps1'])
            self.assertEqual(list(location.iterdir()), [])

    def test_empty_prefix_is_refused(self):
        # Windows PowerShell 5.1 removes empty environment variables, so exercise
        # the resolver directly to test its empty-string contract on every host.
        source = PS_INSTALLER.read_text()
        resolver = source[source.index('function Resolve-InstallPrefix'):source.index('$prefix = if')]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'empty-prefix.ps1'
            script.write_text("$ErrorActionPreference = 'Stop'\n" + resolver +
                              "Resolve-InstallPrefix ''\n", encoding='utf-8-sig')
            result = subprocess.run([shutil.which('pwsh') or 'powershell', '-NoProfile', '-File', str(script)],
                                    capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('PREFIX must name an install directory', result.stderr)

    def test_staged_version_is_silent_and_failures_keep_output(self):
        source = PS_INSTALLER.read_text()
        staged = source[source.index('    try { $stagedVersionOutput'):source.index('    if (Test-Path -LiteralPath $prefix)')]
        installed = source[source.index('    & $exe --version'):source.index('    # Let a new window')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / 'native.ps1'
            native.write_text("'fixture version'; $global:LASTEXITCODE = [int]$env:PROBE_EXIT\n")
            script = root / 'test.ps1'
            script.write_text("$ErrorActionPreference = 'Stop'\n$stageExe = $exe = '" + str(native) + "'\n" + staged + installed)
            shell = shutil.which('pwsh') or 'powershell'
            for code in ['0', '7']:
                result = subprocess.run([shell, '-NoProfile', '-File', str(script)],
                                        env={**os.environ, 'PROBE_EXIT': code}, capture_output=True, text=True)
                if code == '0':
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), 'fixture version')
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Staged vork --version failed', result.stderr)
                    self.assertIn('fixture version', result.stderr)
                    self.assertEqual(result.stdout, '')

    def test_retry_path_is_not_evaluated_as_shell_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "Zoë O'Neil $(throw 'injected') `x`"
            folder.mkdir()
            native = folder / 'native.ps1'
            native.write_text("param([string]$action)\nif ($action -ne 'setup') { exit 1 }; 'native setup'\n", encoding='utf-8-sig')
            source = PS_INSTALLER.read_text()
            quote = source[source.index('function Quote-ShellLiteral'):source.index('$bin = $prefix')]
            script = root / 'test.ps1'
            script.write_text("$ErrorActionPreference = 'Stop'\n" + quote + "\n$exe = @'\n" + str(native) + "\n'@\n" +
                              '$quotedExe = Quote-ShellLiteral $exe\nInvoke-Expression "& $quotedExe setup"\n', encoding='utf-8-sig')
            shell = shutil.which('pwsh') or 'powershell'
            result = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'native setup')

@unittest.skipUnless(os.name == 'nt', 'PowerShell installer is Windows-only')
class PowerShellInstallerTests(unittest.TestCase):
    """VORK_TEST_BINARY (a real vork.exe on a GPU host) enables the full install and warm-up."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ, PREFIX=str(self.root / 'prefix'), USERPROFILE=str(self.root / 'home'),
                        HOME=str(self.root / 'home'), VERSION='test', VORK_DOWNLOAD_BASE=self.root.as_uri())
        self.archive = self.root / 'vork-test-windows-x64.zip'
        binary = os.environ.get('VORK_TEST_BINARY')
        self.fixture('vork.exe', Path(binary).read_bytes() if binary else b'not an executable')

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, member, binary=b'not an executable'):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr(member, binary)
            z.writestr('examples/scene.json', '{}') # export-example-generated: synthetic zip member
            z.writestr('NOTICE', 'notices')
            for name, variable in [('slang', 'VORK_TEST_SLANG'), ('oidn', 'VORK_TEST_OIDN')]:
                if os.environ.get(variable):
                    tool = Path(os.environ[variable])
                    for path in tool.rglob('*'):
                        if path.is_file():
                            z.write(path, name + '/' + path.relative_to(tool).as_posix())
        (self.root / 'SHA256SUMS').write_text(hashlib.sha256(self.archive.read_bytes()).hexdigest() +
                                            '  ' + self.archive.name + '\n')

    def run_installer(self, *args):
        shell = shutil.which('pwsh') or 'powershell'
        return subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(PS_INSTALLER), *args],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    @unittest.skipUnless(all(os.environ.get(k) for k in ('VORK_TEST_BINARY', 'VORK_TEST_SLANG', 'VORK_TEST_OIDN')), 'needs vork.exe, pinned toolchains and GPU')
    def test_global_install_and_setup(self):
        result = self.run_installer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / 'prefix/NOTICE').exists())
        self.assertTrue((self.root / 'prefix/vork.exe').exists())

    def test_corruption_fails_before_install(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'corruption')
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.root / 'prefix/vork.exe').exists())

    def test_missing_checksum_fails(self):
        (self.root / 'SHA256SUMS').write_text('')
        self.assertNotEqual(self.run_installer().returncode, 0)

    def test_duplicate_checksum_fails(self):
        sums = self.root / 'SHA256SUMS'; sums.write_text(sums.read_text() * 2)
        self.assertNotEqual(self.run_installer().returncode, 0)

    def test_traversal_fails(self):
        self.fixture('../vork.exe')
        self.assertNotEqual(self.run_installer().returncode, 0)
        self.assertFalse((self.root / 'prefix/vork.exe').exists())

    def test_dry_run_downloads_and_writes_nothing(self):
        self.env['VORK_DOWNLOAD_BASE'] = 'https://github.com/voyager-so/vork/releases/download/vtest'
        result = self.run_installer('-DryRun')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SHA256SUMS', result.stdout)
        self.assertFalse((self.root / 'prefix').exists())

if __name__ == '__main__':
    unittest.main()
