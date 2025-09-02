// src/ontology/types.ts
export type OntologyItem = { id: string; aliases: string[] };
export type Ontology = {
  treatments: OntologyItem[];
  outcomes: OntologyItem[];
  confounders: OntologyItem[];
  indicators: OntologyItem[];
  baseEstimates: OntologyItem[];
};

export const emptyOntology: Ontology = {
  treatments: [],
  outcomes: [],
  confounders: [],
  indicators: [],
  baseEstimates: [],
};
