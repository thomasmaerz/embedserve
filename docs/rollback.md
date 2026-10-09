# Rollback and recovery

Prepare rollback before changing the live symlink, unit, key, or client configuration.

## Server release

1. Stop new embedding jobs.
2. Point `/opt/embedserve/current` to the previous tested release.
3. Restore the previous `/etc/systemd/system/embedserve.service` if the unit changed.
4. Run `systemctl daemon-reload && systemctl restart embedserve.service`.
5. Verify authenticated health, tags, a synthetic vector, VRAM, and logs.

If dependency pins changed, restore the preserved `/opt/embedserve/venv` artifact before
restarting. Source-only releases may share the existing environment.

## API key

During the bounded rotation window, preserve the prior mode-`0600` files on server and
approved clients. To roll back, atomically restore all prior files, compare fingerprints,
restart server and clients, and test missing/incorrect/valid authentication. Never make
the service temporarily unauthenticated.

## SlackQuery clients

Restore the prior `.env` artifact, file mode, and process definition. Restart the local
or ETL process, run `slackquery embedding-status`, then run representative lexical,
semantic, and hybrid searches. The legacy source copy is not a deployment rollback;
use the preserved deployable release artifact.

## FreeHire client and query endpoint

Stop the `embed` worker and `semantic-agent`, restore the prior FreeHire extension
release and Compose override, and restore `/opt/freehire/embed.env` if its URL or key
changed. Verify lexical search still works before restarting any embedding stage.

For a partially completed E5 generation, preserve a database backup. In one reviewed
transaction, delete `job_semantic_chunks` only for jobs stamped with
`intfloat/multilingual-e5-base-chunked-v1`, clear those jobs' semantic model/hash stamps,
and clear matching live outbox rows. Never relabel vectors or mix model identities. If
the E5 corpus remains intact, the query endpoint can be disabled independently without
deleting vectors.
