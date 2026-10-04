#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { decidePromotion, splitFeedback, trainLocalModel, type FeedbackRecord, type LocalDecisionModel, type PromotionPolicy } from "./index.js";

const [command, ...args] = process.argv.slice(2);
if (!command || !["train", "promote"].includes(command)) {
  console.error("Usage: decision-loop train <feedback.json> <model.json> | promote <candidate.json> <validation.json> [baseline.json] [policy.json]");
  process.exit(1);
}
try {
  if (command === "train") {
    const records = JSON.parse(await readFile(args[0]!, "utf8")) as FeedbackRecord[];
    const split = splitFeedback(records);
    const model = trainLocalModel(split.train);
    await writeFile(args[1]!, `${JSON.stringify({ model, validationIds: split.validation.map(item => item.id), splitFingerprint: split.fingerprint }, null, 2)}\n`, { flag: "wx", mode: 0o600 });
    console.log(JSON.stringify({ modelId: model.id, trainingExamples: model.trainingExamples, validationExamples: split.validation.length, output: args[1] }, null, 2));
  } else {
    const candidatePayload = JSON.parse(await readFile(args[0]!, "utf8")) as LocalDecisionModel | { model: LocalDecisionModel };
    const candidate = "model" in candidatePayload ? candidatePayload.model : candidatePayload;
    const validation = JSON.parse(await readFile(args[1]!, "utf8")) as FeedbackRecord[];
    const baselinePayload = args[2] ? JSON.parse(await readFile(args[2], "utf8")) as LocalDecisionModel | { model: LocalDecisionModel } : undefined;
    const baseline = baselinePayload ? ("model" in baselinePayload ? baselinePayload.model : baselinePayload) : undefined;
    const policy = args[3] ? JSON.parse(await readFile(args[3], "utf8")) as PromotionPolicy : {};
    const result = decidePromotion(candidate, validation, { ...(baseline ? { baseline } : {}), policy });
    console.log(JSON.stringify(result, null, 2));
    if (!result.promote) process.exitCode = 2;
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 2;
}
