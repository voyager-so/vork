import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('shared_build', Path(__file__).parents[1] / 'shared-build.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = dict(commit='a' * 40, toolchain='rust-slang', features=['one'])

    def test_compiler_wrapper_forces_namespace_local_mode_after_cargo_env(self):
        import os
        import subprocess
        fake = self.root / 'fake-sccache'
        fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(log=os.getenv("SCCACHE_LOG"),error=os.getenv("SCCACHE_ERROR_LOG"),mode=os.getenv("SCCACHE_CLIENT_SIDE"),config=os.getenv("SCCACHE_CONF"),args=sys.argv[1:])))\n')
        fake.chmod(0o755)
        env = dict(os.environ, VORK_SCCACHE=str(fake), SCCACHE_LOG='debug', SCCACHE_ERROR_LOG='log', SCCACHE_CLIENT_SIDE='0', SCCACHE_CONF='distributed.toml')
        wrapper = Path(b.__file__).with_name('sccache-client.py')
        result = json.loads(subprocess.check_output([str(wrapper), 'rustc', 'input.rs'], env=env, text=True))
        self.assertEqual(result, dict(log=None, error=None, mode='1', config=os.devnull, args=['rustc', 'input.rs']))

    def test_ancestor_cargo_configuration_changes_identity(self):
        parent = self.root / 'parent'; config = parent / '.cargo/config.toml'
        config.parent.mkdir(parents=True)
        home = self.root / 'cargo-home'
        before = b.cargo_config_fingerprints(home, [parent / 'nested'])
        config.write_text('[build]\nrustflags=["-Copt-level=1"]\n')
        after = b.cargo_config_fingerprints(home, [parent / 'nested'])
        self.assertNotEqual(before, after)
        config.write_text('[build]\nrustflags=["-Copt-level=2"]\n')
        self.assertNotEqual(after, b.cargo_config_fingerprints(home, [parent / 'nested']))

    def test_producer_freshness_isolated_by_identity(self):
        source, target = b.build_paths(self.root, self.identity['commit'], self.identity)
        changed = dict(self.identity, commit='b' * 40)
        other_source, other_target = b.build_paths(self.root, changed['commit'], changed)
        self.assertNotEqual(source, other_source)
        self.assertNotEqual(target, other_target)
        for field, value in [('toolchain', 'new-rust'), ('features', ['two'])]:
            changed = dict(self.identity, **{field: value})
            same_source, other_target = b.build_paths(self.root, changed['commit'], changed)
            self.assertEqual(source, same_source)
            self.assertNotEqual(target, other_target)

    def artifact(self):
        for name in ('vork', 'vork-slang-filesystem'):
            (self.root / name).write_bytes(b'compiled')
        receipt = dict(identity=self.identity, files={n: b.digest(self.root / n) for n in ('vork', 'vork-slang-filesystem')})
        (self.root / 'receipt.json').write_text(json.dumps(receipt))
        return receipt

    def test_wrong_commit_toolchain_features_are_refused(self):
        self.artifact()
        b.validate(self.root, self.identity)
        for field in ('commit', 'toolchain', 'features'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity'):
                b.validate(self.root, dict(self.identity, **{field: 'different'}))

    def test_corruption_and_symlinks_are_refused(self):
        self.artifact()
        (self.root / 'vork').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            b.validate(self.root, self.identity)
        (self.root / 'vork').unlink()
        (self.root / 'vork').symlink_to('vork-slang-filesystem')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            b.validate(self.root, self.identity)

    def test_missing_companion_is_refused(self):
        receipt = self.artifact()
        del receipt['files']['vork-slang-filesystem']
        (self.root / 'receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'required'):
            b.validate(self.root, self.identity)

    def test_archive_rejects_traversal_links_and_directories(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('vork', tarfile.SYMTYPE), ('dir', tarfile.DIRTYPE)]:
            archive = self.root / 'bad.tar'
            with tarfile.open(archive, 'w') as tf:
                member = tarfile.TarInfo(name); member.type = kind; member.linkname = '/etc/passwd'
                tf.addfile(member)
            with self.assertRaisesRegex(ValueError, 'flat regular'):
                b.unpack(archive, self.root / 'extract')
        self.assertFalse((self.root.parent / 'escape').exists())

    def test_round_trip(self):
        self.artifact()
        archive = self.root / 'good.tar'
        with tarfile.open(archive, 'w') as tf:
            for name in ('vork', 'vork-slang-filesystem', 'receipt.json'):
                tf.add(self.root / name, arcname=name)
        dest = self.root / 'unpacked'
        b.unpack(archive, dest)
        b.validate(dest, self.identity)

    def test_toolchain_content_additions_and_edits_change_key(self):
        inputs = self.root / 'tools'; inputs.mkdir()
        p = inputs / 'compiler'; p.write_bytes(b'first')
        memo = self.root / 'memo'
        first = b.tree_digest([('tools', inputs)], memo)
        self.assertEqual(first, b.tree_digest([('tools', inputs)], memo))
        p.write_bytes(b'other')
        second = b.tree_digest([('tools', inputs)], memo)
        self.assertNotEqual(first, second)
        (inputs / 'library').write_bytes(b'library')
        self.assertNotEqual(second, b.tree_digest([('tools', inputs)], memo))

    def test_lock_waits_and_deletes_only_owned_generation_on_failure(self):
        cloud = b.Cloud('test')
        with patch.object(cloud, 'request', side_effect=[None, b'{"generation":"123"}', b'']) as request, patch.object(b.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'build failed'):
                with cloud.lock('key', 100):
                    raise RuntimeError('build failed')
        self.assertEqual(request.call_args.args, ('DELETE', 'locks/key'))
        self.assertEqual(request.call_args.kwargs, {'generation': '123'})

    def test_busy_lock_never_takes_over_or_deletes(self):
        cloud = b.Cloud('test')
        with patch.object(cloud, 'request', return_value=None) as request:
            with self.assertRaisesRegex(RuntimeError, 'no automatic stale-lock takeover'):
                with cloud.lock('key', 0):
                    self.fail('acquired someone else\'s lock')
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[0], 'POST')

class ShaderSharingTests(unittest.TestCase):
    def test_shader_round_trip_and_corruption_refusal(self):
        spec = importlib.util.spec_from_file_location('shared_shader', Path(__file__).parents[1] / 'shared-shader-cache.py')
        s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir()
            key = 'a' * 64; entry = source / key; entry.mkdir()
            (entry / 'ready').touch(); (entry / '0').write_bytes(b'spirv'); (entry / '1').write_bytes(b'reflection')
            stage = root / 'stage'; stage.mkdir()
            manifest = s.export_entry(source, key, stage)
            dest = root / 'dest'; dest.mkdir()
            (stage / '0').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                s.import_entry(dest, key, stage, manifest)
            self.assertFalse((dest / key).exists())
            (stage / '0').write_bytes(b'spirv')
            self.assertTrue(s.import_entry(dest, key, stage, manifest))
            self.assertEqual((dest / key / '1').read_bytes(), b'reflection')

class PublicationTests(unittest.TestCase):
    def test_local_hit_does_not_wait_for_unrelated_producer(self):
        # A slow build of commit B must not delay fetching already-ready commit A.
        import sys
        import fcntl
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); work = root / 'work'; work.mkdir()
            expected = {'commit': 'a' * 40}
            key = 'a' * 40 + '/' + b.hashlib.sha256(b.encoded(expected)).hexdigest()
            final = root / 'store' / key; final.mkdir(parents=True)
            for name in ('vork', 'vork-slang-filesystem'):
                (final / name).write_bytes(b'executable')
            (final / 'receipt.json').write_bytes(b.encoded(dict(identity=expected,
                files={n: b.digest(final / n) for n in ('vork', 'vork-slang-filesystem')})))
            args = ['shared-build', 'HEAD', '--work', str(work), '--store', str(root / 'store'),
                    '--bucket', 'test', '--slang', str(root / 'slang')]
            with (work / 'producer.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with patch.object(sys, 'argv', args), patch.object(b, 'run', return_value='a' * 40), \
                     patch.object(b, 'identity', return_value=expected), patch.object(b, 'Cloud') as cloud, \
                     patch.object(b.fcntl, 'flock', side_effect=AssertionError('hit acquired a producer lock')), \
                     patch('builtins.print') as output:
                    b.main()
                cloud.assert_not_called(); output.assert_called_once_with(final)


class ConcurrentProducerTests(unittest.TestCase):
    def test_two_clients_never_hold_the_same_cloud_lock(self):
        import threading
        state_lock = threading.Lock()
        objects = {}
        active = 0
        maximum = 0
        errors = []
        first_entered = threading.Event()
        release_first = threading.Event()

        def request(method, name, data=None, generation=None):
            with state_lock:
                if method == 'POST':
                    if name in objects:
                        return None
                    objects[name] = '7'
                    return b'{"generation":"7"}'
                if objects.get(name) == generation:
                    del objects[name]
                return b''

        def worker(first):
            nonlocal active, maximum
            try:
                client = b.Cloud('test')
                client.request = request
                with client.lock('same-key', 5):
                    with state_lock:
                        active += 1; maximum = max(maximum, active)
                    if first:
                        first_entered.set()
                        if not release_first.wait(2):
                            raise RuntimeError('test synchronization timeout')
                    with state_lock:
                        active -= 1
            except Exception as error:
                errors.append(error)

        with patch.object(b.time, 'sleep', side_effect=lambda _: threading.Event().wait(0.01)):
            one = threading.Thread(target=worker, args=(True,)); one.start()
            self.assertTrue(first_entered.wait(2))
            two = threading.Thread(target=worker, args=(False,)); two.start()
            release_first.set(); one.join(3); two.join(3)
        self.assertFalse(one.is_alive() or two.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(maximum, 1)
        self.assertEqual(objects, {})


class SourceProvenanceTests(unittest.TestCase):
    def test_dirty_producer_source_never_builds(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp); (work / 'sources' / ('a' * 40)).mkdir(parents=True)
            a = SimpleNamespace(work=work)
            with patch.object(b, 'run', return_value=' M tracked.rs'), patch.object(b.subprocess, 'run') as compile:
                with self.assertRaisesRegex(RuntimeError, 'dirty'):
                    b.build(a, 'a' * 40, {}, work / 'stage')
                compile.assert_not_called()

    def test_source_head_change_during_build_never_publishes(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp); (work / 'sources' / ('a' * 40)).mkdir(parents=True)
            a = SimpleNamespace(work=work, cargo='cargo', features='', no_default_features=False, all_features=False)
            with patch.object(b, 'run', side_effect=['', 'b' * 40]), patch.object(b, 'environment', return_value={}), patch.object(b, 'cache_ready'), \
                 patch.object(b.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), patch.object(b, 'identity') as fingerprint:
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    b.build(a, 'a' * 40, {}, work / 'stage')
                fingerprint.assert_not_called()
            self.assertFalse((work / 'stage').exists())


class CacheIsolationTests(unittest.TestCase):
    def test_wrong_daemon_is_refused_without_restart(self):
        from types import SimpleNamespace
        a = SimpleNamespace(sccache='sccache', bucket='ours')
        with patch.object(b, 'run', return_value='{"cache_location":"Local disk: existing"}') as command:
            with self.assertRaisesRegex(RuntimeError, 'do not restart'):
                b.cache_ready(a, {}, Path('.'))
        self.assertEqual(command.call_count, 1)
        self.assertEqual(command.call_args.args[0], ['sccache', '--show-stats', '--stats-format=json'])


class FreshMachineTests(unittest.TestCase):
    def test_first_request_authenticates_during_first_half_hour_of_uptime(self):
        cloud = b.Cloud('test')
        with patch.object(b.time, 'monotonic', return_value=5) as clock, \
             patch.object(b, 'run', return_value='unit-test-token') as credential, \
             patch.object(b.urllib.request, 'urlopen') as http:
            http.return_value.__enter__.return_value.read.return_value = b'{}'
            self.assertEqual(cloud.get('manifest'), {})
            self.assertEqual(cloud.get('manifest'), {})
            self.assertEqual(credential.call_count, 1)
            clock.return_value = 2000
            self.assertEqual(cloud.get('manifest'), {})
            self.assertEqual(credential.call_count, 2)


if __name__ == '__main__':
    unittest.main()
