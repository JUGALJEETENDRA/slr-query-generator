"""Package only the standalone runtime and documentation; verify every ZIP member."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
target = root.parent / f'litsync-ieee-collector-v{manifest["version"]}.zip'
files = [root / 'manifest.json', root / 'README.md', root / 'VALIDATION.md']
files += sorted((root / 'src').glob('*.js'))
files += sorted((root / 'popup').glob('*'))
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file, file.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == {file.relative_to(root).as_posix() for file in files}
    for file in files:
        assert archive.read(file.relative_to(root).as_posix()) == file.read_bytes()
digest = hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix('.zip.sha256').write_text(f'{digest}  {target.name}\n', encoding='ascii')
print(f'Validated {len(files)} package files: {target}\nSHA256: {digest}')
