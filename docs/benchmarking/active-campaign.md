# Active NVMe campaign — accelerated 2026-10-09

[**Live campaign status and ETA**](http://172.16.2.20:3000/d/kvbench-campaign)
updates every 10 seconds and follows phase/run transitions automatically.
The information below is the launch record; use the dashboard for current progress.

**Status as checked at 2026-10-09 18:29 UTC: running, in accelerated filler preparation.**
There is no qualified endurance result yet. This page records the launch;
runtime state and final summaries are in the artifacts below.

## Identity and supervision

- Unit: `kvbench-waf-5d497ce9.service`, on dev, running as `wvaske`.
- Frozen source revision: `5d497ce9d1110a0be96e4513ad4c990470a178bc`.
- Detached checkout:
  `/mnt/development/wvaske/kvbench-runs/source-5d497ce9d1110a0be96e4513ad4c990470a178bc`.
- Configuration: that checkout's `examples/drive-compare.yaml`.
- Physical drive: Dell/Micron 6500 RI FIPS 30.7 TB, serial `2337448A0113`,
  `/dev/nvme6n1`; test filesystem `/mnt/work`, workload root `/mnt/work/kvbench`.
- Run artifacts: `/mnt/development/wvaske/kvbench-runs` (off the test drive).
- Preparation ID: `prepare-20261009T182701-c56ab3ac`.
- Preparation telemetry: `<artifacts>/<preparation-ID>/samples.jsonl`.
- Owned filler: `/mnt/work/kvbench/<preparation-ID>/`; manifest updated per 8 GiB file.

The unit's whole process group is limited to 4 GiB memory, with a 3 GiB
memory-high threshold and no swap. That accounting includes reclaimable file
cache. CPU is limited to eight core-equivalents, with low I/O priority and a
3 GB/s read/write limit on the namespace. Filler has a 2,048 MiB/s pacing ceiling,
uses fresh SFC64 random words, and fsyncs each 256 MiB. Actual sustained host
writes measured about **0.95–1.01 GB/s**, versus about 0.132 GB/s before the change
(roughly 7–8×). The observed cgroup memory peak was about 3 GiB, below the hard
limit; host/NAND counters and validity were checked through the live Grafana API.

## Preserved preparation and faster workload

The original `kvbench-waf-4aa32c30.service` was stopped intentionally for this
upgrade. Its approximately 1 TB of filler and all telemetry remain under
`prepare-20261009T160447-e8d0b3d8`; the new campaign counts existing occupied
space and only supplies the remainder. Nothing was erased or overwritten.

Both cache arms now use the same fast, bounded SFC64 payload generator. The
CPU microbenchmark improved from approximately 81 to 2,041 MiB/s (about 25×).
HiCache's modeled host pages are reused within their configured pool instead
of allocating and faulting new tensors for every page.

The configured operating point is 8×B300, 200,000 prefill and 20,000 decode
tokens/s, 16 clients, 32 active slots, 8,192-token prefill batches and 512-token
outputs. Full-speed mode skips compute sleeps; these configured rates are not
measured B300 performance. Actual throughput remains limited by generation,
manager behavior, filesystem and device I/O.

The real-drive pilot at
`/mnt/development/wvaske/kvbench-runs/speed-pilot/20261009T182154-9828055e`
completed both arms with valid host/NAND samples. File-operation traces measured
about **653 MB/s LMCache** (up to four concurrent writes) and **471 MB/s HiCache**
(one synchronous writer). An earlier bounded pilot fell short of its 8 GiB host
target because LMCache declined some page admissions under the small host-tier
budget; its evidence was retained. The successful functional pilot used a 4 GiB
observed-host target. **Endurance requirements were not reduced**. These pilots
remain `qualified:false` and establish no comparative endurance-WAF conclusion.

## Automatic sequence

1. Write real random filler to reach 85% filesystem occupancy. At the accelerated
   restart the drive was about 16.8% occupied, requiring roughly 21 TB additional
   data. At the observed rate, preparation is approximately **6 hours**, rather
   than the original 46-hour estimate, if that rate persists.
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
systemctl status kvbench-waf-5d497ce9.service
journalctl -u kvbench-waf-5d497ce9.service -f

# Stops only this campaign; leaves existing data, filler and evidence intact.
sudo systemctl stop kvbench-waf-5d497ce9.service
```

[Grafana dashboard](http://172.16.2.20:3000/d/kvbench-nvme-waf). Select the
preparation ID now; the comparison creates a new run ID after preparation.
The independent `kvbench-nvme.service` continues monitoring if the campaign stops.

Before accelerated launch: **705 tests passed, one skipped**, Ruff/mypy passed, strict
MkDocs and sdist/wheel builds passed. Real-drive smoke validation included both
adapters, raw host/NAND counters, all Grafana queries and zero-drop block traces.
See [protocol and bounded observations](drive-waf.md); those smoke runs were
explicitly unqualified for endurance conclusions.
