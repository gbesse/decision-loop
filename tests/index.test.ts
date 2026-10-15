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

test('prototype-like user labels and tokens remain finite after JSON model reload',()=>{
 const source=[{id:'one',text:'constructor __proto__',label:'constructor',reviewedAt:'2026-10-01'}, {id:'two',text:'ordinary other words',label:'__proto__',reviewedAt:'2026-10-01'}];
 const model=JSON.parse(JSON.stringify(trainLocalModel(source)));
 const result=predict(model,'constructor __proto__');assert.ok(Number.isFinite(result.confidence));assert.equal(Object.keys(result.probabilities).length,2);
});
test('promotion blocks training overlap unknown provenance invalid policies and baseline-free gain',()=>{
 const split=splitFeedback(records),model=trainLocalModel(split.train);
 assert.equal(decidePromotion(model,split.train).promote,false);
 const legacy={...model};delete legacy.trainingRecordIds;assert.equal(decidePromotion(legacy,split.validation).promote,false);
 for(const value of [NaN,Infinity,-1])assert.throws(()=>decidePromotion(model,split.validation,{policy:{maxAccuracyRegression:value}}),/promotion policy/);
 assert.equal(decidePromotion(model,split.validation,{policy:{minAccuracyGain:.1}}).promote,false);
});
test('promotion receipt binds complete model and validation content even when metrics are unchanged',()=>{
 const split=splitFeedback(records),model=trainLocalModel(split.train,{id:'same',createdAt:'2026-10-01'});
 const first=decidePromotion(model,split.validation),second=decidePromotion({...model,createdAt:'2026-10-02'},split.validation);
 assert.notEqual(first.receipt,second.receipt);assert.notEqual(first.candidateFingerprint,second.candidateFingerprint);
 const modified=split.validation.map(record=>({...record,text:record.text+' '}));assert.notEqual(first.validationFingerprint,decidePromotion(model,modified).validationFingerprint);
});
