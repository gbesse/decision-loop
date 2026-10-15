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

## Bound promotion evidence / Preuves de promotion liées / Pruebas vinculadas

EN: New models retain training record IDs. Promotion blocks known training/validation ID overlap and legacy artifacts without these IDs; a requested gain needs a baseline. Finite policy bounds are validated before evaluation. Receipts now expose hashes of the complete model, baseline and validation records, rather than relying on IDs and aggregate metrics alone. Prototype-like tokens/labels remain ordinary user data even after JSON reload. IDs are caller-declared and do not prove independent collection, reviewer identity or absence of duplicated text; hashes are unsigned. Promotion remains a recommendation, with no automatic production deployment or SOTA claim.

FR : Les nouveaux modèles conservent les ID d’entraînement. La promotion bloque un chevauchement avec validation et les anciens artefacts sans ID ; un gain demandé exige une référence. Les bornes de politique sont validées avant évaluation. Les reçus exposent désormais les hashes du modèle complet, de la référence et des dossiers de validation, au lieu des seuls ID/métriques. Les tokens/labels de type prototype restent des données ordinaires après rechargement JSON. Les ID déclarés ne prouvent ni collecte indépendante, ni identité du relecteur, ni absence de textes dupliqués ; hashes non signés. La promotion reste une recommandation, sans déploiement automatique ni revendication SOTA.

ES: Modelos nuevos conservan ID de entrenamiento. Promoción bloquea solapamiento con validación y artefactos antiguos sin ID; una mejora solicitada exige referencia. Valida límites antes de evaluar. Recibos muestran hashes del modelo completo, referencia y registros de validación, no solo ID/métricas. Tokens/etiquetas similares a prototipos siguen siendo datos ordinarios tras recargar JSON. ID declarados no prueban recogida independiente, identidad del revisor ni ausencia de textos duplicados; hashes sin firma. Promoción sigue siendo recomendación, sin despliegue automático ni afirmación SOTA.
