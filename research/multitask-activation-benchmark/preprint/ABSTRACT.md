An agent workflow may need several judgments about the same input: toxicity, sentiment, spam,
permission to act, or what to remember. IntentProbe runs one small frozen local language model and
shares its hidden states across independently trained activation probes. Each task can choose its
own layers, token pooling, normalization, classifier and threshold. The model does not have to
generate a separate verdict for every check.

This preprint explains the architecture, the readout contracts and the implementation that readers
can reproduce. A CPU sidecar can work beside a hosted agent without access to the hosted model's
hidden states. In the completed Qwen2.5-0.5B experiment, one forward plus one head averaged
68.63 ms and the same shared path with 16 heads averaged 73.34 ms across 80 held-out texts per K
(paired mean ratio 1.069; 95% interval 1.056–1.078). The measured suite covers toxicity, jailbreak,
spam, sentiment, emotion and conversational pragmatics. The project also releases tool-poisoning,
action-authorization and memory-write probes with local scorers.

The report connects the design to existing probing systems and gives code, model and benchmark
references. The general registry and hot-loading interface are specified next steps. Further
comparisons with cached judges and compact classifier fleets can extend this initial account.
