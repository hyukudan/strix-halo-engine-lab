"""Private metadata-keyed model SHA256 cache; never copied to public results."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


class ModelHashes:
    def __init__(self, directory, mode, hasher):
        if mode not in ('full', 'cached', 'skip'):
            raise ValueError('verify-models must be full, cached or skip')
        self.mode, self.hasher = mode, hasher
        directory = Path(directory).resolve()
        fallback = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'strix-bench'
        self.locations = [directory / '.bench-sha256-cache.json',
                          fallback / (hashlib.sha256(str(directory).encode()).hexdigest() + '.json')]
        self.entries = {}
        for location in reversed(self.locations):
            try:
                self.entries.update(json.loads(location.read_text(encoding='utf-8'))['entries'])
            except (OSError, ValueError, KeyError, TypeError):
                pass

    @staticmethod
    def key(path):
        path = Path(path).resolve()
        st = path.stat()
        return [str(path), st.st_size, st.st_mtime_ns, st.st_ino]

    def digest(self, path):
        if self.mode == 'skip':
            return None
        key = self.key(path)
        entry = self.entries.get(key[0], {})
        digest = entry.get('sha256')
        if self.mode == 'cached' and entry.get('key') == key and isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest):
            return digest
        digest = self.hasher(path)
        if self.key(path) != key:
            raise ValueError('model changed while hashing: ' + Path(path).name)
        self.entries[key[0]] = {'key': key, 'sha256': digest}
        return digest

    def save(self):
        if self.mode == 'skip':
            return
        for location in self.locations:
            temporary = None
            try:
                location.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=location.parent,
                                                 prefix='.bench-cache-', delete=False) as f:
                    temporary = Path(f.name)
                    json.dump({'schema_version': 1, 'entries': self.entries}, f)
                os.replace(temporary, location)
                return
            except OSError:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise ValueError('cannot write model hash cache in model directory or XDG cache')
