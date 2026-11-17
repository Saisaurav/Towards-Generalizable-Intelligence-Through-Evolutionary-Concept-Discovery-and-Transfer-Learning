# Concept Bank frontend update

The Concept Bank is now a minimal point field driven by the real concept-bank API response.

- Removed the hard-coded demo ontology and its animated fake connections/pulses.
- Removed the second canvas visualization and all fabricated chart/activation-rate content.
- Concepts render as small, unlabeled points; active concepts use the lime accent.
- Selecting a point opens the inspector with the stored expression, primitive dependencies, and available admission/source/target metadata.
- Added search filtering and an inspector close control.
- Removed the long bank-entry list and active-concepts list below the graph.
- The point positions are a deterministic visual scatter, not a semantic embedding; no relationships or similarity are implied.

Validation: inline JavaScript syntax checks passed; browser smoke test rendered all 28 concepts from the supplied concept bank, opened the inspector on point selection, and closed it successfully with no page errors.
