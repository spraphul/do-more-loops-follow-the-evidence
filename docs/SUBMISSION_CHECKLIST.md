# Submission upload checklist

Before exposing this repository to reviewers:

1. Create a new public repository under an identity-neutral account or an
   approved anonymous-repository service. Do not fork the private project.
2. Use this repository's `main` branch and its existing single anonymous
   commit. Do not copy private remotes, issues, pull requests, or tags.
3. Run `make verify` from a fresh clone. Confirm that continuous integration
   passes on Python 3.11.
4. Run `make reproduce` once in the release environment and retain the console
   log outside the repository for the authors' records.
5. Check the public repository page for account names, avatars, organization
   links, sponsorship links, or profile metadata that could identify an
   author.
6. Link the anonymous repository from the submission's reproducibility
   statement. Do not link a private Overleaf project or private experiment
   tracker.
7. Download the hosted source archive and rerun `python
   tools/audit_anonymity.py` after extraction.

After the review process, author and citation metadata can be added in a new
commit without rewriting the anonymous review snapshot.

