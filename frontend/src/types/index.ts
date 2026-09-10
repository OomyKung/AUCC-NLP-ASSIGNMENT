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
