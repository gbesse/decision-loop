import test from "node:test";
import assert from "node:assert/strict";
import { decidePromotion, evaluateModel, predict, selectHardExamples, splitFeedback, trainLocalModel, type FeedbackRecord } from "../src/index.js";

const records: FeedbackRecord[] = [
  { id: "1", text: "read project file", label: "allow", reviewedAt: "2026-10-01T00:00:00Z", partition: "train" },
  { id: "2", text: "inspect source safely", label: "allow", reviewedAt: "2026-10-01T00:00:00Z", partition: "train" },
  { id: "3", text: "delete all files", label: "deny", reviewedAt: "2026-10-01T00:00:00Z", partition: "train" },
  { id: "4", text: "destroy project data", label: "deny", reviewedAt: "2026-10-01T00:00:00Z", partition: "train" },
  { id: "5", text: "read source file", label: "allow", reviewedAt: "2026-10-01T00:00:00Z", partition: "validation" },
  { id: "6", text: "delete project data", label: "deny", reviewedAt: "2026-10-01T00:00:00Z", partition: "validation" },
];

test("creates a deterministic leakage-resistant split", () => {
  const split = splitFeedback(records);
  assert.equal(split.train.length, 4);
  assert.equal(split.validation.length, 2);
  assert.equal(split.fingerprint.length, 64);
});

test("trains and evaluates a versioned local classifier", () => {
  const split = splitFeedback(records);
  const model = trainLocalModel(split.train, { id: "candidate", createdAt: "2026-10-04T00:00:00Z" });
  assert.equal(predict(model, "delete files").label, "deny");
  assert.equal(evaluateModel(model, split.validation).accuracy, 1);
  assert.equal(decidePromotion(model, split.validation, { policy: { minValidationCases: 2 } }).promote, true);
});

test("prioritizes disagreements and low-confidence feedback", () => {
  const selected = selectHardExamples([
    { ...records[0]!, previousLabel: "deny", previousConfidence: .9 },
    { ...records[1]!, previousLabel: "allow", previousConfidence: .1 },
  ], 1);
  assert.equal(selected[0]?.id, "1");
});

test("enforces a requested minimum promotion gain", () => {
  const split = splitFeedback(records);
  const model = trainLocalModel(split.train, { id: "same" });
  const decision = decidePromotion(model, split.validation, { baseline: model, policy: { minAccuracyGain: .01 } });
  assert.equal(decision.promote, false);
  assert.match(decision.reasons[0]!, /gain/);
});
