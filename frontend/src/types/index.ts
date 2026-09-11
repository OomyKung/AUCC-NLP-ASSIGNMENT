/**
 * Types mirroring the FastAPI response models.
 * Kept in one file so a backend schema change has a single place to land.
 */

export type SentimentSlug = 'positive' | 'neutral' | 'negative'

export interface Taxonomy {
  slug: string
  thai: string
  english: string
  color: string
  color_dark?: string
}

export interface Keyword {
  word: string
  score: number
}

export interface NamedEntity {
  text: string
  label: string
}

export interface NewsListItem {
  id: number
  title: string
  summary: string | null
  topic: string
  topic_confidence: number
  sentiment: SentimentSlug
  sentiment_confidence: number
  keywords: Keyword[]
  source: string
  url: string | null
  published_at: string
  source_type: string
  message_count: number | null
}

export interface NLPAnalysis {
  cleaned_text: string | null
  tokens: string[]
  filtered_tokens: string[]
  token_count: number
  unique_token_count: number
  stopword_removed_count: number
  topic_probabilities: Record<string, number>
  sentiment_probabilities: Record<string, number>
  entities: NamedEntity[]
  keyword_scores: Keyword[]
  sentences: string[]
  model_versions: Record<string, string>
  pipeline_version: string
  processing_ms: number
}

export interface NewsDetail extends NewsListItem {
  content: string
  created_at: string
  window_start: string | null
  window_end: string | null
  analysis: NLPAnalysis | null
}

export interface NewsPage {
  items: NewsListItem[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface ChatMessage {
  id: number
  author: string | null
  text: string
  published_at: string
  sentiment: SentimentSlug | null
  sentiment_confidence: number | null
}

export interface LabelCount {
  key: string
  label: string
  label_en: string
  color: string
  count: number
}

export interface SentimentSlice extends LabelCount {
  percentage: number
}

export interface TopicSentimentRow {
  topic: string
  label: string
  label_en: string
  color: string
  positive: number
  neutral: number
  negative: number
  total: number
}

export interface KeywordCount {
  word: string
  count: number
  score: number
}

export interface Statistics {
  total_news: number
  positive: number
  neutral: number
  negative: number
  most_common_topic: string | null
  most_common_topic_label: string | null
  most_common_topic_count: number
  total_messages: number
  scored_messages: number
  noise_messages: number
  total_streams: number
  chat_windows: number
  articles: number
  by_topic: LabelCount[]
  by_sentiment: SentimentSlice[]
  topic_sentiment: TopicSentimentRow[]
  top_keywords: KeywordCount[]
  avg_topic_confidence: number
  avg_sentiment_confidence: number
}

export type TrendGranularity = 'daily' | 'weekly' | 'monthly'

export interface TrendPoint {
  period: string
  count: number
  positive: number
  neutral: number
  negative: number
}

export interface TrendResponse {
  granularity: TrendGranularity
  points: TrendPoint[]
}

export interface Stream {
  id: number
  video_id: string
  url: string
  title: string | null
  channel: string | null
  collector: string
  is_live: boolean
  message_count: number
  window_count: number
  first_message_at: string | null
  last_message_at: string | null
}

export interface BackendStatus {
  stage: string
  requested: string
  active: string
  trained: boolean
  note: string
}

export interface Health {
  status: string
  app: string
  version: string
  python: string
  nlp_backends: Record<string, string>
  llm_summarizer_configured: boolean
}

export interface AnalyzeResult {
  topic: string
  topic_confidence: number
  sentiment: SentimentSlug
  sentiment_confidence: number
  summary: string
  keywords: Keyword[]
  entities: NamedEntity[]
  topic_probabilities: Record<string, number>
  sentiment_probabilities: Record<string, number>
  cleaned_text: string
  tokens: string[]
  filtered_tokens: string[]
  token_count: number
  unique_token_count: number
  stopword_removed_count: number
  sentences: string[]
  processing_ms: number
  model_versions: Record<string, string>
  news_id: number | null
}

export interface IngestResult {
  video_id: string
  title: string | null
  channel: string | null
  is_live: boolean
  collected: number
  stored: number
  duplicates: number
  windows_created: number
  messages_scored: number
  messages_skipped_noise: number
  snapshot: string | null
}

export interface SnapshotInfo {
  file: string
  video_id: string
  title: string | null
  channel: string | null
  message_count: number | null
  size_kb: number
}

export interface NewsFilters {
  search?: string
  topic?: string
  sentiment?: string
  source_type?: string
  date_from?: string
  date_to?: string
  min_confidence?: number
  sort?: 'newest' | 'oldest' | 'confidence' | 'messages'
  page?: number
  page_size?: number
}

export interface PerClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

export interface CrossValidation {
  folds: number
  scoring: string
  scores: number[]
  mean: number
  std: number
}

export interface ModelMetrics {
  accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
  precision_weighted: number
  recall_weighted: number
  f1_weighted: number
  per_class: Record<string, PerClassMetrics>
  confusion_matrix: { labels: string[]; matrix: number[][] }
  support: number
  algorithm: string
  trained_at?: string
  cross_validation?: CrossValidation | null
  /** Blend weight, present only on the blended model. */
  alpha?: number
}

/** Keys of the models evaluate.py scores on the held-out split. */
export type ModelKey = 'trained' | 'blend' | 'baseline'

export interface TaskEvaluation {
  labels: string[]
  models: Partial<Record<ModelKey, ModelMetrics>> & { baseline: ModelMetrics }
  random_baseline_accuracy: number
  trained_vs_baseline_f1_macro?: number
  label_names?: Record<string, { thai: string; english: string; color: string }>
  /** Which scored model is actually serving requests. */
  active_model?: ModelKey
  /** The configured NLP backend name, for display. */
  configured_backend?: string
}

export interface Evaluation {
  available: boolean
  reason?: string
  how_to?: string[]
  generated_at: string
  dataset: {
    path: string
    documents: number
    topics: Record<string, number>
    sentiments: Record<string, number>
  }
  split: {
    test_size: number
    random_state: number
    stratified: boolean
    note: string
  }
  tasks: Record<string, TaskEvaluation>
}


/* -------------------------------------------------------------------------
 * Spoken content: what the newsreader said, as opposed to the chat.
 * ---------------------------------------------------------------------- */

/** One story detected inside a programme's audio. */
export interface NewsSegment {
  id: number
  position: number
  start_ms: number
  end_ms: number
  duration_ms: number
  /** H:MM:SS label as a viewer reads it. */
  timecode: string
  headline: string
  summary: string
  topic: string
  topic_label: string
  topic_color: string
  topic_confidence: number
  sentiment: string
  sentiment_label: string
  sentiment_confidence: number
  keywords: string[]
  entities: { text: string; label: string }[]
  /** Bounded excerpt of what was said, for the expander. */
  transcript_text_preview: string
  /** Which signals put a boundary here, so a split can be explained. */
  boundary_reasons: string[]
  boundary_confidence: number
  /** Deep link straight to the moment this story starts. */
  youtube_url: string
  /** Frame captured from the video at that moment. */
  frame_url: string | null
}

export interface BroadcastTranscript {
  source: string
  language: string
  duration_ms: number
  cue_count: number
  character_count: number
}

export interface Programme {
  video_id: string
  title: string | null
  channel: string | null
  url: string
  transcript: BroadcastTranscript | null
  segment_count: number
  segments: NewsSegment[]
}

export interface SegmentPage {
  total: number
  limit: number
  offset: number
  items: NewsSegment[]
}

export interface AnalyseVideoResult {
  video_id: string
  transcript_source: string
  duration_ms: number
  cue_count: number
  segment_count: number
  frames_captured: number
  frame_note: string
}
