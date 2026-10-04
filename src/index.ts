import { createHash, randomUUID } from "node:crypto";

export interface FeedbackRecord {
  id: string;
  text: string;
  label: string;
  reviewedAt: string;
  reviewer?: string;
  previousLabel?: string;
  previousConfidence?: number;
  partition?: "train" | "validation";
  metadata?: Record<string, string>;
}

export interface DatasetSplit { train: FeedbackRecord[]; validation: FeedbackRecord[]; fingerprint: string; }

export interface LocalDecisionModel {
  schemaVersion: 1;
  id: string;
  createdAt: string;
  algorithm: "multinomial-naive-bayes";
  labels: string[];
  vocabulary: string[];
  classDocuments: Record<string, number>;
  classTokens: Record<string, number>;
  tokenCounts: Record<string, Record<string, number>>;
  trainingFingerprint: string;
  trainingExamples: number;
  alpha: number;
}

export interface LocalPrediction { label: string; probabilities: Record<string, number>; confidence: number; margin: number; }
export interface ModelMetrics { cases: number; accuracy: number; macroF1: number; logLoss: number; confusion: Record<string, Record<string, number>>; }
export interface PromotionPolicy { minAccuracyGain?: number; maxAccuracyRegression?: number; maxLogLossRegression?: number; minValidationCases?: number; }
export interface PromotionDecision { promote: boolean; reasons: string[]; candidate: ModelMetrics; baseline?: ModelMetrics; receipt: string; }

function assert(condition: unknown, message: string): asserts condition { if (!condition) throw new Error(message); }
function round(value: number): number { return Math.round(value * 1000000) / 1000000; }
function canonical(value: unknown): unknown { return Array.isArray(value) ? value.map(canonical) : value && typeof value === "object" ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, child]) => [key, canonical(child)])) : value; }
function fingerprint(value: unknown): string { return createHash("sha256").update(JSON.stringify(canonical(value))).digest("hex"); }
function tokenize(text: string): string[] { return text.normalize("NFKC").toLocaleLowerCase("en-US").match(/[\p{L}\p{N}_-]{2,}/gu)?.slice(0, 10_000) ?? []; }

export function validateFeedback(records: FeedbackRecord[]): void {
  assert(records.length > 0, "Feedback cannot be empty");
  const ids = new Set<string>();
  for (const record of records) {
    assert(record.id?.trim() && !ids.has(record.id), `Duplicate or missing feedback id: ${record.id}`);
    ids.add(record.id);
    assert(record.text?.trim(), `${record.id}: text is required`);
    assert(record.label?.trim(), `${record.id}: label is required`);
    assert(Number.isFinite(Date.parse(record.reviewedAt)), `${record.id}: reviewedAt must be an ISO date`);
    if (record.previousConfidence !== undefined) assert(record.previousConfidence >= 0 && record.previousConfidence <= 1, `${record.id}: previousConfidence must be between zero and one`);
  }
}

export function splitFeedback(records: FeedbackRecord[], options: { validationFraction?: number; salt?: string } = {}): DatasetSplit {
  validateFeedback(records);
  const fraction = options.validationFraction ?? .2;
  assert(fraction > 0 && fraction < 1, "validationFraction must be between zero and one");
  const train: FeedbackRecord[] = [];
  const validation: FeedbackRecord[] = [];
  for (const record of records) {
    const target = record.partition ?? (Number.parseInt(createHash("sha256").update(`${options.salt ?? "decision-loop-v1"}:${record.id}`).digest("hex").slice(0, 8), 16) / 0xffffffff < fraction ? "validation" : "train");
    (target === "validation" ? validation : train).push(structuredClone(record));
  }
  assert(train.length > 0 && validation.length > 0, "Split needs at least one training and one validation record; set explicit partitions for small datasets");
  return { train, validation, fingerprint: fingerprint(records.map(({ id, label, reviewedAt, partition }) => ({ id, label, reviewedAt, partition }))) };
}

export function trainLocalModel(records: FeedbackRecord[], options: { alpha?: number; id?: string; createdAt?: string } = {}): LocalDecisionModel {
  validateFeedback(records);
  const alpha = options.alpha ?? 1;
  assert(Number.isFinite(alpha) && alpha > 0, "alpha must be positive");
  const labels = [...new Set(records.map(record => record.label))].sort();
  assert(labels.length >= 2, "Training requires at least two labels");
  const vocabulary = new Set<string>();
  const classDocuments: Record<string, number> = Object.fromEntries(labels.map(label => [label, 0]));
  const classTokens: Record<string, number> = Object.fromEntries(labels.map(label => [label, 0]));
  const tokenCounts: Record<string, Record<string, number>> = Object.fromEntries(labels.map(label => [label, {}]));
  for (const record of records) {
    classDocuments[record.label] = (classDocuments[record.label] ?? 0) + 1;
    for (const token of tokenize(record.text)) {
      vocabulary.add(token);
      classTokens[record.label] = (classTokens[record.label] ?? 0) + 1;
      tokenCounts[record.label]![token] = (tokenCounts[record.label]![token] ?? 0) + 1;
    }
  }
  assert(vocabulary.size > 0, "Training text produced no tokens");
  return {
    schemaVersion: 1, id: options.id ?? randomUUID(), createdAt: options.createdAt ?? new Date().toISOString(), algorithm: "multinomial-naive-bayes",
    labels, vocabulary: [...vocabulary].sort(), classDocuments, classTokens, tokenCounts,
    trainingFingerprint: fingerprint(records.map(({ id, text, label }) => ({ id, text, label }))), trainingExamples: records.length, alpha,
  };
}

export function predict(model: LocalDecisionModel, text: string): LocalPrediction {
  assert(model.schemaVersion === 1 && model.algorithm === "multinomial-naive-bayes", "Unsupported model artifact");
  const tokens = tokenize(text).filter(token => model.vocabulary.includes(token));
  const totalDocuments = Object.values(model.classDocuments).reduce((a, b) => a + b, 0);
  const scores: Record<string, number> = {};
  for (const label of model.labels) {
    let score = Math.log(((model.classDocuments[label] ?? 0) + model.alpha) / (totalDocuments + model.alpha * model.labels.length));
    const denominator = (model.classTokens[label] ?? 0) + model.alpha * model.vocabulary.length;
    for (const token of tokens) score += Math.log(((model.tokenCounts[label]?.[token] ?? 0) + model.alpha) / denominator);
    scores[label] = score;
  }
  const max = Math.max(...Object.values(scores));
  const exponentials = Object.fromEntries(Object.entries(scores).map(([label, score]) => [label, Math.exp(score - max)]));
  const total = Object.values(exponentials).reduce((a, b) => a + b, 0);
  const probabilities = Object.fromEntries(Object.entries(exponentials).map(([label, value]) => [label, round(value / total)]));
  const ranked = Object.entries(probabilities).sort(([labelA, a], [labelB, b]) => b - a || labelA.localeCompare(labelB));
  return { label: ranked[0]![0], probabilities, confidence: ranked[0]![1], margin: round(ranked[0]![1] - (ranked[1]?.[1] ?? 0)) };
}

export function evaluateModel(model: LocalDecisionModel, records: FeedbackRecord[]): ModelMetrics {
  validateFeedback(records);
  const labels = [...new Set([...model.labels, ...records.map(record => record.label)])].sort();
  const confusion: Record<string, Record<string, number>> = Object.fromEntries(labels.map(label => [label, Object.fromEntries(labels.map(predicted => [predicted, 0]))]));
  let correct = 0;
  let logLoss = 0;
  for (const record of records) {
    const prediction = predict(model, record.text);
    confusion[record.label]![prediction.label] = (confusion[record.label]![prediction.label] ?? 0) + 1;
    if (prediction.label === record.label) correct++;
    logLoss -= Math.log(Math.max(1e-12, prediction.probabilities[record.label] ?? 0));
  }
  const f1s = labels.map(label => {
    const tp = confusion[label]?.[label] ?? 0;
    const fp = labels.filter(other => other !== label).reduce((sum, other) => sum + (confusion[other]?.[label] ?? 0), 0);
    const fn = labels.filter(other => other !== label).reduce((sum, other) => sum + (confusion[label]?.[other] ?? 0), 0);
    return 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0;
  });
  return { cases: records.length, accuracy: round(correct / records.length), macroF1: round(f1s.reduce((a, b) => a + b, 0) / f1s.length), logLoss: round(logLoss / records.length), confusion };
}

export function selectHardExamples(records: FeedbackRecord[], limit = 100): FeedbackRecord[] {
  assert(Number.isInteger(limit) && limit > 0, "limit must be positive");
  return [...records].sort((a, b) => {
    const disagreementA = a.previousLabel && a.previousLabel !== a.label ? 1 : 0;
    const disagreementB = b.previousLabel && b.previousLabel !== b.label ? 1 : 0;
    return disagreementB - disagreementA || (a.previousConfidence ?? 0) - (b.previousConfidence ?? 0) || Date.parse(b.reviewedAt) - Date.parse(a.reviewedAt) || a.id.localeCompare(b.id);
  }).slice(0, limit);
}

export function decidePromotion(candidate: LocalDecisionModel, validation: FeedbackRecord[], options: { baseline?: LocalDecisionModel; policy?: PromotionPolicy } = {}): PromotionDecision {
  const candidateMetrics = evaluateModel(candidate, validation);
  const baselineMetrics = options.baseline ? evaluateModel(options.baseline, validation) : undefined;
  const policy = options.policy ?? {};
  const reasons: string[] = [];
  if (validation.length < (policy.minValidationCases ?? 1)) reasons.push(`validation has ${validation.length} cases; need ${policy.minValidationCases}`);
  if (baselineMetrics) {
    const gain = candidateMetrics.accuracy - baselineMetrics.accuracy;
    if (gain < (policy.minAccuracyGain ?? 0) && baselineMetrics.accuracy - candidateMetrics.accuracy > (policy.maxAccuracyRegression ?? 0)) reasons.push(`accuracy regression ${round(baselineMetrics.accuracy - candidateMetrics.accuracy)} is not allowed`);
    if (candidateMetrics.logLoss - baselineMetrics.logLoss > (policy.maxLogLossRegression ?? 0)) reasons.push(`log-loss regression ${round(candidateMetrics.logLoss - baselineMetrics.logLoss)} is not allowed`);
  }
  const core = { candidateId: candidate.id, baselineId: options.baseline?.id ?? null, validationFingerprint: fingerprint(validation.map(({ id, label }) => ({ id, label }))), policy, candidate: candidateMetrics, baseline: baselineMetrics ?? null, reasons };
  return { promote: reasons.length === 0, reasons, candidate: candidateMetrics, ...(baselineMetrics ? { baseline: baselineMetrics } : {}), receipt: fingerprint(core) };
}
