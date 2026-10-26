# Offline verification boundary

Offline install/update/refusal fixtures use a synthetic local Git upstream.

## Reproduce

`python3 -m unittest discover -s tests -v`

Compatibility matrix: local Git fixture + current overlay = covered; real ClaudeClaw/npm/plugins/gateway = NOT_RUN. The no-rsync fallback and automatic upstream network pull are not certified by the fixture. Back up the whole runtime before applying an update; no uninstall is supplied.

## Rollback

Review the commit diff and use a new checkout of the pre-change commit for source rollback. Keep user data and live installations untouched. A source rollback is not permission to restore unsafe network exposure or bypass approval.
