### CI: a pull request's superseded run is cancelled; publishing needs a green CI run at the tagged commit

`ci.yml` cancels a pull request's superseded run; push runs keep one group each, so every main commit keeps its own
run. The conflict-marker guard runs on pushes to `main` and on every pull request head. `publish.yml` refuses to upload
unless `ci.yml` has a successful push or dispatch run at the tagged commit, read from the checkout, so a manual
dispatch checks the tag and not the branch it was started from. It fails closed: a `gh` failure or a count
that isn't an integer refuses. `kernel/test_ci_workflow_shape.py` pins all of it.
