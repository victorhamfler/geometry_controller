#!/usr/bin/env -S npx tsx

import process from "node:process";
import { createLcmDatabaseConnection, closeLcmConnection } from "/home/victo/.openclaw/extensions/lossless-claw/src/db/connection.ts";
import {
  applyDoctorCleaners,
  getDoctorCleanerFilterIds,
  scanDoctorCleaners,
  type DoctorCleanerId,
} from "/home/victo/.openclaw/extensions/lossless-claw/src/plugin/lcm-doctor-cleaners.ts";

type CliOptions = {
  dbPath: string;
  apply: boolean;
  vacuum: boolean;
  json: boolean;
  filterId?: DoctorCleanerId;
};

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    dbPath: process.env.LCM_DB_PATH?.trim() || "/home/victo/.openclaw/lcm.db",
    apply: false,
    vacuum: false,
    json: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg) continue;

    if (arg === "--apply") {
      opts.apply = true;
      continue;
    }
    if (arg === "--vacuum") {
      opts.vacuum = true;
      continue;
    }
    if (arg === "--json") {
      opts.json = true;
      continue;
    }
    if (arg === "--db" && argv[i + 1]) {
      opts.dbPath = String(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg.startsWith("--db=")) {
      opts.dbPath = arg.slice("--db=".length);
      continue;
    }
    if (arg === "--filter" && argv[i + 1]) {
      opts.filterId = argv[i + 1] as DoctorCleanerId;
      i += 1;
      continue;
    }
    if (arg.startsWith("--filter=")) {
      opts.filterId = arg.slice("--filter=".length) as DoctorCleanerId;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  if (opts.filterId && !getDoctorCleanerFilterIds().includes(opts.filterId)) {
    throw new Error(
      `Invalid filter id: ${opts.filterId}. Allowed: ${getDoctorCleanerFilterIds().join(", ")}`,
    );
  }

  return opts;
}

function printHelp() {
  console.log(
    [
      "lossless_doctor_clean_runner.ts",
      "",
      "Usage:",
      "  npx tsx /home/victo/.openclaw/workspace/scripts/lossless_doctor_clean_runner.ts [options]",
      "",
      "Options:",
      "  --db <path>            LCM db path (default: /home/victo/.openclaw/lcm.db)",
      "  --filter <id>          One cleaner filter id",
      "  --apply                Apply deletion (creates backup first)",
      "  --vacuum               VACUUM after apply",
      "  --json                 JSON output",
      "  -h, --help             Show help",
      "",
      `Filter IDs: ${getDoctorCleanerFilterIds().join(", ")}`,
    ].join("\n"),
  );
}

function printTextScan(scan: ReturnType<typeof scanDoctorCleaners>) {
  console.log("[lossless-doctor-clean] scan");
  console.log(`  distinct_conversations=${scan.totalDistinctConversations}`);
  console.log(`  distinct_messages=${scan.totalDistinctMessages}`);
  for (const filter of scan.filters) {
    console.log(
      `  - ${filter.id}: conversations=${filter.conversationCount} messages=${filter.messageCount}`,
    );
    for (const ex of filter.examples.slice(0, 3)) {
      const preview = ex.firstMessagePreview ? ` preview=${JSON.stringify(ex.firstMessagePreview)}` : "";
      console.log(
        `      example conversation_id=${ex.conversationId} session_key=${JSON.stringify(ex.sessionKey)} msg_count=${ex.messageCount}${preview}`,
      );
    }
  }
}

function printTextApply(result: ReturnType<typeof applyDoctorCleaners>) {
  if (result.kind === "unavailable") {
    console.log("[lossless-doctor-clean] apply unavailable");
    console.log(`  reason=${result.reason}`);
    return;
  }
  console.log("[lossless-doctor-clean] apply done");
  console.log(`  deleted_conversations=${result.deletedConversations}`);
  console.log(`  deleted_messages=${result.deletedMessages}`);
  console.log(`  vacuumed=${result.vacuumed}`);
  console.log(`  backup_path=${result.backupPath}`);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const db = createLcmDatabaseConnection(opts.dbPath);

  try {
    const filterIds = opts.filterId ? [opts.filterId] : undefined;
    const scan = scanDoctorCleaners(db, filterIds);

    if (opts.apply) {
      const apply = applyDoctorCleaners(db, {
        databasePath: opts.dbPath,
        filterIds,
        vacuum: opts.vacuum,
      });

      if (opts.json) {
        console.log(
          JSON.stringify(
            {
              mode: "apply",
              dbPath: opts.dbPath,
              filterId: opts.filterId ?? null,
              scan,
              apply,
            },
            null,
            2,
          ),
        );
      } else {
        printTextScan(scan);
        printTextApply(apply);
      }
      return;
    }

    if (opts.json) {
      console.log(
        JSON.stringify(
          {
            mode: "scan",
            dbPath: opts.dbPath,
            filterId: opts.filterId ?? null,
            scan,
          },
          null,
          2,
        ),
      );
      return;
    }

    printTextScan(scan);
    console.log("[lossless-doctor-clean] dry run only (no deletion)");
  } finally {
    closeLcmConnection(db);
  }
}

main().catch((err) => {
  console.error(`[lossless-doctor-clean] ERROR: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
