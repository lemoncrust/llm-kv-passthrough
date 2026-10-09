# Active NVMe campaign — started 2026-10-09

**Status as checked at 2026-10-09 16:06 UTC: running, in filler preparation.**
There is no qualified endurance result yet. This page records the launch;
runtime state and final summaries are in the artifacts below.

## Identity and supervision

- Unit: `kvbench-waf-4aa32c30.service`, on dev, running as `wvaske`.
- Frozen source revision: `4aa32c308db15e9a6ea0ab82223669a7a8c6c4a2`.
- Detached checkout:
  `/mnt/development/wvaske/kvbench-runs/source-4aa32c308db15e9a6ea0ab82223669a7a8c6c4a2`.
- Configuration: that checkout's `examples/drive-compare.yaml`.
- Physical drive: Dell/Micron 6500 RI FIPS 30.7 TB, serial `2337448A0113`,
  `/dev/nvme6n1`; test filesystem `/mnt/work`, workload root `/mnt/work/kvbench`.
- Run artifacts: `/mnt/development/wvaske/kvbench-runs` (off the test drive).
- Preparation ID: `prepare-20261009T160447-e8d0b3d8`.
- Preparation telemetry: `<artifacts>/<preparation-ID>/samples.jsonl`.
- Owned filler: `/mnt/work/kvbench/<preparation-ID>/`; manifest updated per 8 GiB file.

The unit's whole process group is limited to 4 GiB memory, with a 3 GiB
memory-high threshold and no swap. That accounting includes reclaimable file
cache. CPU is limited to two core-equivalents, with low I/O priority and a
512 MB/s read/write limit on the namespace. Filler is additionally paced at
128 MiB/s. The observed initial cgroup memory peak was about 3 GiB, below the
hard limit; both host/NAND counter streams were valid and reached Grafana.

## Automatic sequence

1. Write real random filler to reach 85% filesystem occupancy. At launch the
   drive was 13.4% occupied, requiring roughly 22 TB additional data. At the
   configured pace preparation is approximately 46 hours.
2. Qualify the occupied drive and check idle background-write rate.
3. Run LMCache → HiCacheFile → HiCacheFile → LMCache, with a fresh directory and
   process for each arm. Each arm requires one observed host drive fill of
   preconditioning, then three measured fills plus the duration requirement.
4. Drain manager/OS writes and collect the idle NAND tail for each phase.

The full ABBA workload requires approximately 491.5 TB of observed host writes,
in addition to initial filler. It can take many days. A duration timeout,
counter/identity failure, sampling gap or occupancy violation fails the run;
it cannot become a completed/qualified result merely by reaching a deadline.

## Inspect or stop

```bash
systemctl status kvbench-waf-4aa32c30.service
journalctl -u kvbench-waf-4aa32c30.service -f

# Stops only this campaign; leaves existing data, filler and evidence intact.
sudo systemctl stop kvbench-waf-4aa32c30.service
```

[Grafana dashboard](http://172.16.2.20:3000/d/kvbench-nvme-waf). Select the
preparation ID now; the comparison creates a new run ID after preparation.
The independent `kvbench-nvme.service` continues monitoring if the campaign stops.

Before launch: **689 tests passed, one skipped**, Ruff/mypy passed, strict
MkDocs and sdist/wheel builds passed. Real-drive smoke validation included both
adapters, raw host/NAND counters, all Grafana queries and zero-drop block traces.
See [protocol and bounded observations](drive-waf.md); those smoke runs were
explicitly unqualified for endurance conclusions.
