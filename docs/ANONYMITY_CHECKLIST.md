# Double-blind release checklist

- No author names or affiliations.
- No private Git history or inherited remote.
- No personal email address, user name, host name, or machine path.
- No authentication token, environment file, checkpoint cache, or model
  weight.
- No licensed natural-question prose, raw answer continuation, or source item
  identifier.
- No PDF author or creator field that identifies a person or institution.
- Anonymous commit author and committer metadata.
- Public release digests separated from private-source digests.
- Secondary analyses retain their original evidence status.

The `make verify` target runs `python tools/audit_anonymity.py`. The audit
returns an error when it finds credential-shaped strings or known private path
forms.
