# CPU deployment of the fixed GOSDT/CART diagnostic

Prepared 31 October 2026. Scientific protocol: [spec 12](specs/12_gosdt_structure_learning.md).
No real-data fit or scheduler job was run during local preparation. Commands are
executed manually in an authenticated terminal. The submission script creates a
synthetic check and a conditional experiment, which runs only if the check succeeds
and its input/code/runtime receipt still matches. Cluster enforcement has not yet
been verified by local tests.

## Verified context and operational request

Supplied terminal output confirms login01, default Python 3.13.5, account dm_warm_up,
normal QoS, allgroups (including CPU nodes), cgroup v2 and task/cgroup. The cluster
checkout was at 8d23e33. No Python 3.11/3.12 or environment manager was found on PATH.
The two CQ summaries exist, but the verdict references are absent. Neither file
size nor presence establishes source identity.

Use a fresh detached checkout, preserving the existing repository and run files.
Transfer the eight-file archive, not the complete evaluations tree. The archive is
5.85 MiB, SHA256 ea6f00713577d7303cab88bbcd7398c59461e2856a751f3cd387904209b6bb27.
Both batch modes require its per-file checksums.

The proposed allocation is one node, two CPUs and 8 GiB, with no GPU request. The
driver and sequential worker can each use one CPU. Scheduler ceilings are one hour
for the synthetic check and 48 hours for the conditional experiment. These are not
duration predictions and do not change the 60-second native solver budget,
five-second handling grace or scientific grid. The script first uses sbatch
--test-only; account/partition admission remains unverified until that succeeds.

## Runtime and memory enforcement

Setup creates a private Python 3.12 environment and a separate worker dependency
directory. It preserves system Python and shell startup files. Scientific driver
libraries match the local implementation environment; GOSDT 1.0.4 and its numerical
libraries match the worker checks. The actual Python patch version and driver
package inventory are recorded. Setup is package/download work only, refuses to
overwrite an existing runtime and requires network access to distributors.

Each native Linux fit is a numbered srun step inside the existing allocation.
Slurm's MiB granularity rounds 4,000,000,000 bytes downward to 3,814 MiB:
3,999,268,864 bytes, a difference below 0.02%. Before numerical imports or fitting,
the worker requires an effective cgroup memory.max equal to that rounded request.
A missing, larger or unexpectedly smaller cap fails the check. There is no
uncapped fallback. Linux charged RAM includes page cache and differs from Windows
private-commit accounting. The actual cap, hierarchy, counters, RSS/USS observations
and worker peak RSS are recorded.

The driver checks cgroup headroom and host free RAM. Existing free-RAM gates and
the one-GiB reserve remain active. Native time_limit stays 60 seconds; a parent
watchdog and Linux kernel alarm bound handling time. Cancellation targets only
the worker's numbered step. Each new allocation repeats its runtime probe.

The synthetic check verifies effective kernel memory limits and records native
peak usage, then runs all six configurations on three partitions of 96 invented
rows with 45 Boolean predicates. A separate one-second native control records
TIMEOUT/incumbent behavior without changing the real 60-second budget. No benchmark
label chooses feasibility settings. All 18 grid fits must return valid trees with
exact held-out replay. Missing caps, invalid grid solutions or incompatible runtime
prevent approval and the dependent experiment. Deliberate OOM is not required for
approval; real memory failures remain explicit in supervision records. Neither the
invented grid nor local tests certify worst-case combined B1/solver memory or time.

## Manual procedure

1. Push the validated implementation and transfer the scratch input archive to
   the cluster project parent. No bulk evaluations are published to Git.
2. Fetch that commit in the existing cluster repository and create a fresh detached
   sibling checkout with git worktree add. Use the exact commit, not an unfrozen branch.
3. Extract with tar --keep-old-files into the fresh checkout, then verify all eight
   hashes with sha256sum -c input_checksums.sha256. Preserve existing files on error.
4. Run bash serving/gosdt_cluster_setup.sh from the fresh checkout. This downloads
   packages and prepares the runtime; it does not submit a job or fit data.
5. Run bash serving/gosdt_cluster_submit.sh to submit the synthetic check and the
   conditional experiment. It prints both IDs and log paths. The experiment uses
   afterok and kill-on-invalid-dep, and checks the matching approval receipt itself.

Monitor with squeue -u "$USER" and read runs/slurm/gosdt-*.out from that checkout.
Approval, synthetic records and driver freeze stay under scratch; experiment
artifacts go to evaluations/gosdt_schemes2_diagnostic_20261030. If setup or a job
fails, preserve logs and outputs. Do not alter scientific settings to obtain a
successful result. An incomplete run remains incomplete. Resume is a separate
explicit action with unchanged snapshots, not automatic resubmission.

## Primary technical references

- [Slurm srun](https://slurm.schedmd.com/srun.html): exact step resources, memory
  requests, signals and return codes.
- [Slurm cgroup configuration](https://slurm.schedmd.com/cgroup.conf.html):
  enforcement depends on configuration; a request alone is not proof.
- [Slurm sbatch](https://slurm.schedmd.com/sbatch.html): test-only submission and
  dependent jobs.
- [uv Python management](https://docs.astral.sh/uv/guides/install-python/): private
  installation of a requested Python version using python-build-standalone.
- [GOSDT 1.0.4 files](https://pypi.org/project/gosdt/1.0.4/#files): Linux CPython
  3.12 wheel.
