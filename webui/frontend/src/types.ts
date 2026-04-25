/**
 * Shared TypeScript types between the WebUI frontend and the FastAPI
 * backend. Kept hand-written (no code-gen) because the backend surface
 * is small and iteration is faster.
 */

export interface TaskSchema {
  name: string;
  display_name: string;
  features: string[];
  feature_types: Record<string, string>;
  categorical_values: Record<string, unknown[]>;
  design_bounds: Record<string, [number, number]>;
  target_columns: Record<string, string>;
  all_product_columns: string[];
  product_names: Record<string, string>;
  default_target: string;
  /** "builtin" for hard-coded Python tasks, "project" for editable JSON projects. */
  source?: "builtin" | "project";
}

// ---------------------------------------------------------------------------
// Declarative project (dynamic TaskBase definition)
// ---------------------------------------------------------------------------
export interface FeatureSpec {
  name: string;
  type: "continuous" | "integer";
  lo: number;
  hi: number;
  unit?: string | null;
  display_name?: string | null;
}

export interface TargetSpec {
  short_name: string;
  column: string;
  display_name?: string | null;
  unit?: string | null;
  is_competing: boolean;
}

export interface ProjectSpec {
  name: string;
  display_name: string;
  description: string;
  features: FeatureSpec[];
  targets: TargetSpec[];
  default_target: string;
  notes: string;
}

export interface ProjectsListResponse {
  projects: ProjectSpec[];
  builtins: string[];
}

export interface RunConfig {
  task: string;
  data_path: string;
  candidates_path: string | null;
  target_product: string | null;
  top_k: number;
  beta: number;
  beta_schedule: string;
  beta_delta: number;
  acq_strategy: string;
  qnei_mc_samples: number;
  kernel_type: string;
  h2_penalty_weight: number;
  skip_feature_selection: boolean;
  strict_training_schema: boolean;
  pre_fill_before_choice: boolean;
  seed: number | null;
  device: string;
  iterations: number;
  interactive: boolean;
  kabo_mode: boolean;
  lambda_p: number;
  lambda_k: number;
  lambda_v: number;
  expert_prior_file: string | null;
  diversity_weight: number;
  pe_budget: number;
  generate_candidates_n: number;
  prefer_file_candidates: boolean;
  discrete_strategy: string;
}

export type RunStatus =
  | "idle"
  | "pending"
  | "running"
  | "done"
  | "error"
  | "aborted";

export interface StatusResponse {
  status: RunStatus;
  run_id?: string | null;
  pending_prompt?: PromptEvent | null;
  error?: string | null;
}

// ---------------------------------------------------------------------------
// Events pushed over SSE
// ---------------------------------------------------------------------------
export type KaboEvent =
  | LogEvent
  | PromptEvent
  | RecommendationsEvent
  | BestFoundEvent
  | VisualizationEvent
  | RunLifecycleEvent;

export interface LogEvent {
  type: "log";
  ts: number;
  level: string;
  message: string;
}

export interface RecommendationsEvent {
  type: "recommendations";
  ts: number;
  iteration: number;
  target_column: string;
  target_name: string;
  top_n: number;
  recommendations: Recommendation[];
  selected_features: string[];
  all_features: string[];
}

export interface Recommendation {
  rank: number;
  idx: number;
  acq_value: number;
  source: string;
  features: Record<string, { value: number | null; origin: string }>;
}

export interface BestFoundEvent {
  type: "best_found";
  ts: number;
  target_column: string;
  target_name: string;
  best_value: number | null;
  products: Record<string, number | null>;
  features: Record<string, number | null>;
}

export interface VizPanelImage {
  image: string; // data:image/png;base64,...
  iteration: number;
  dims?: [string, string];
  n_train?: number;
  n_candidates?: number;
}

export interface VisualizationEvent {
  type: "visualization";
  ts: number;
  iteration: number;
  target_column: string;
  target_name: string;
  selected_features: string[];
  gp_landscape: VizPanelImage | null;
  pca_projection: VizPanelImage | null;
}

export interface RunLifecycleEvent {
  type: "run_started" | "run_completed" | "run_failed";
  ts: number;
  run_id?: string;
  config?: Record<string, unknown>;
  task?: string;
  error?: string;
  traceback?: string;
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------
export type PromptKind =
  | "candidate_choice"
  | "manual_candidate"
  | "nonselected_features"
  | "product_yields"
  | "raw_input";

export interface PromptEventBase {
  type: "prompt";
  ts: number;
  prompt_id: number;
  kind: PromptKind;
}

export interface CandidateChoicePrompt extends PromptEventBase {
  kind: "candidate_choice";
  top_indices: number[];
  n_total: number;
}

export interface ManualCandidatePrompt extends PromptEventBase {
  kind: "manual_candidate";
  features: string[];
  bounds: Record<string, [number, number]>;
}

export interface NonselectedFeaturesPrompt extends PromptEventBase {
  kind: "nonselected_features";
  features: string[];
  bounds: Record<string, [number, number]>;
}

export interface ProductYieldsPrompt extends PromptEventBase {
  kind: "product_yields";
  target_column: string;
  target_name: string;
  products: Array<{
    short: string;
    column: string;
    is_target: boolean;
    display: string;
  }>;
}

export interface RawInputPrompt extends PromptEventBase {
  kind: "raw_input";
  prompt_text: string;
}

export type PromptEvent =
  | CandidateChoicePrompt
  | ManualCandidatePrompt
  | NonselectedFeaturesPrompt
  | ProductYieldsPrompt
  | RawInputPrompt;

// ---------------------------------------------------------------------------
// Historical runs
// ---------------------------------------------------------------------------
export interface ArchivedRun {
  run_id: string;
  output_dir: string;
  status?: string;
  started_at?: number;
  finished_at?: number;
  config?: RunConfig;
  metadata?: Record<string, unknown>;
  error?: string | null;
  has_data_updated?: boolean;
  has_feature_importances?: boolean;
}

export interface FileEntry {
  name: string;
  path: string;
  size: number;
  mtime: number;
}
