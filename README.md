# Decision Loop

Decision Loop turns reviewed decisions into a versioned local classifier with
reproducible holdouts, hard-example selection and guarded promotion. It ships a
small multinomial Naive Bayes learner as an inspectable baseline; teams can use
the same artifact and promotion contracts for a fine-tuned head or LoRA.

```sh
npm install @gbesse/decision-loop
decision-loop train examples/feedback.json ./models/candidate.json
```

The output file is created exclusively with mode `600` and is never overwritten.
Each artifact records its training fingerprint, algorithm, vocabulary, label
counts and creation time. Promotion compares accuracy and log loss on the same
reviewed validation set and produces a content-addressed receipt.

```ts
import { splitFeedback, trainLocalModel, decidePromotion } from "@gbesse/decision-loop";

const { train, validation } = splitFeedback(feedback);
const candidate = trainLocalModel(train);
const decision = decidePromotion(candidate, validation, {
  baseline,
  policy: { maxAccuracyRegression: 0, maxLogLossRegression: 0.02 },
});
```

Feedback text can contain confidential data. Training is local and has no
network dependency, but users remain responsible for consent, retention,
redaction and representative independent validation. The built-in classifier
is a baseline, not a safety boundary.

MIT licensed. Independent of TypeSafe, OpenAI and model vendors.
