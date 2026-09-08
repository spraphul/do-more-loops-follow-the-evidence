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

Run `python tools/audit_anonymity.py` immediately before uploading the
repository. The audit deliberately fails closed on credential-shaped strings
and known private path forms.

