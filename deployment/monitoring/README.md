# NVMe WAF monitoring

The read-only sampler collects standard NVMe SMART host writes and OCP SMART
physical-media writes every **5 seconds**. It appends exact raw integer counters
and calculated deltas to JSONL, then atomically publishes the latest Influx line
for the existing native Telegraf agent. Grafana uses the `influxdb-telegraf`
Flux datasource and `telegraf` bucket.

## Deployed on dev (2026-10-09)

- `kvbench-nvme.service`: user `wvaske`, 256 MiB memory limit, read-only `sudo -n
  nvme` commands targeting `/dev/nvme6`, occupancy of `/mnt/work`.
- `/var/lib/kvbench-nvme/samples.jsonl`: raw counters, units implicit in schema,
  timestamps, device identity, interval and run WAF, validity and errors.
- `/var/lib/kvbench-nvme/latest.influx`: atomically replaced metric sample.
- `/etc/telegraf/telegraf.d/kvbench-nvme.conf`: 5-second ingestion.
- `/etc/logrotate.d/kvbench-nvme`: seven daily logs. The sampler opens the file
  for each sample, so rename rotation does not lose an open stream.
- Dashboard provisioned on lp-01 at
  `/mnt/configs/configs-grafana-provisioning/dashboards/kvbench-nvme-waf.json`.
- [Live dashboard](http://172.16.2.20:3000/d/kvbench-nvme-waf).

All seven Flux queries were executed against live data. Counter, rate,
occupancy, drive-fill and validity panels returned data. WAF correctly remained
absent while host-write deltas were zero, even when NAND increased due to idle
background activity. This check is telemetry validation, not an endurance result.

## Install elsewhere

Adapt the paths, user, controller and mount in the service. The service user
needs permission for the specific read-only `nvme id-ctrl`, `nvme smart-log`
and `nvme ocp smart-add-log` commands (with JSON output); do not grant the
collector write/format/sanitize commands. Install the service, Telegraf drop-in
and logrotate file to the corresponding system directories, create the data
directory owned by the service user, enable/start the service, and reload
Telegraf with SIGHUP. Put the dashboard JSON in Grafana's provisioned directory.

No controller is discovered or selected implicitly. Unsupported OCP schemas,
missing NAND counters, command failures, identity changes, resets and sampling
gaps generate invalid samples rather than fabricated NAND bytes or zero WAF.
Original JSONL counters retain 128-bit precision; dashboard numeric fields are
floating point, with deltas and ratios calculated first using exact integers.

For a standalone bounded check:

```bash
python -m kvbench.benchmark.nvme --device /dev/nvme6 --mount /mnt/work \
  --sudo --interval 5 --samples 12 --output /tmp/nvme-check.jsonl
```

The dashboard's window WAF divides **sums of byte deltas**, not averages of
individual ratios. Cumulative WAF includes idle GC after the baseline. Any
counter reset/gap invalidates cumulative WAF until a new collector run starts.
Zero host deltas omit interval WAF but preserve NAND deltas and rates.
Host/NAND counters are controller-wide: other writers affect them. Endurance
experiments additionally qualify single-device backing, namespaces and occupancy.

To stop just this monitoring addition: stop/disable `kvbench-nvme.service`,
remove its Telegraf drop-in and reload Telegraf. Existing monitoring is independent.
