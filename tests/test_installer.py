import os, pathlib, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class Installer(unittest.TestCase):
    def test_local_upstream_preserves_user_files_and_env(self):
        with tempfile.TemporaryDirectory() as td:
            home=pathlib.Path(td); upstream=home/'upstream'; upstream.mkdir()
            env={**os.environ,'HOME':td,'CLAUDECLAW_UPSTREAM_URL':str(upstream)}
            def git(*args): subprocess.run(['git','-C',str(upstream),*args],env=env,check=True,capture_output=True)
            git('init'); (upstream/'README.md').write_text('Synthetic host; not a real runtime.')
            git('add','.'); git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture')
            target=home/'runtime'
            def run(): return subprocess.run(['bash',str(ROOT/'setup.sh'),str(target)],env=env,capture_output=True,text=True)
            p=run(); self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            user=target/'user-owned.md'; user.write_text('preserve')
            config=target/'.env'; config.write_text('SYNTHETIC_REFERENCE=local-only\n')
            p=run(); self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            self.assertEqual(user.read_text(),'preserve')
            self.assertEqual(config.read_text(),'SYNTHETIC_REFERENCE=local-only\n')
    def test_refuse_non_git_target(self):
        with tempfile.TemporaryDirectory() as td:
            p=subprocess.run(['bash',str(ROOT/'setup.sh'),td],env={**os.environ,'HOME':td,'CLAUDECLAW_UPSTREAM_URL':'https://invalid.invalid/not-used'},capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('not a git checkout',p.stderr)
            self.assertEqual(list(pathlib.Path(td).iterdir()),[])
if __name__=='__main__': unittest.main()
