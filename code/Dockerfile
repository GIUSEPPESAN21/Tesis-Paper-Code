# CPU-only, matching this thesis' actual training/evaluation environment exactly --
# torch.cuda.is_available() was False throughout (PreRegistration_FINAL.md Amendment 2).
# No GPU/CUDA base image is used because none of this thesis' results were produced
# with one; installing a CUDA image here would document an environment nobody validated.
FROM python:3.11.9-slim

# Base image already ships close to the exact Python patch version this thesis used
# (3.11.9). ortools/torch/scipy CPU wheels for linux/amd64+py3.11 are pre-built, so no
# compiler toolchain is installed -- keeps the image small and the build fast/stable.

WORKDIR /app

# Dependency manifests copied first so `pip install` is Docker-layer-cached across
# code-only changes -- rebuilding after editing envs/models/experiments doesn't
# re-download ~1GB of torch/ortools every time.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The rest of the source tree. See .dockerignore: checkpoints, raw per-row result
# CSVs, __pycache__, and both venvs are excluded on purpose -- this image captures the
# *environment*, not the multi-hundred-MB results tree. Mount that in at `docker run`
# time (see README.md's "Option B -- Docker" section) so a rebuild never silently goes
# stale relative to whatever experiments/results the reader actually has on their host.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Verify the install is sane at build time, not just at first `docker run` -- a fast,
# non-CP-SAT-heavy subset (full suite is ~7 min and not worth paying at every build).
RUN pytest -m "not slow" -q

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
