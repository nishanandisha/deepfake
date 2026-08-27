import type { QueueItem } from "@/lib/types";

const now = Date.now();
const hoursAgo = (h: number) => new Date(now - h * 3_600_000).toISOString();

// Static seed data so the "recent scans" sidebar doesn't look empty on first load.
export const QUEUE_SEED: QueueItem[] = [
  { sampleId: "seed_1", fileName: "press_briefing_clip.mp4", verdict: "authentic", deepfakeScore: 0.06, createdAt: hoursAgo(0.4) },
  { sampleId: "seed_2", fileName: "protest_livestream_export.mp4", verdict: "uncertain", deepfakeScore: 0.42, createdAt: hoursAgo(1.1) },
  { sampleId: "seed_3", fileName: "interview_segment_09.mp4", verdict: "deepfake", deepfakeScore: 0.88, createdAt: hoursAgo(2.3) },
  { sampleId: "seed_4", fileName: "campaign_statement.mp4", verdict: "authentic", deepfakeScore: 0.11, createdAt: hoursAgo(3.7) },
  { sampleId: "seed_5", fileName: "voicemail_forward.mp3", verdict: "uncertain", deepfakeScore: 0.51, createdAt: hoursAgo(5.2) },
  { sampleId: "seed_6", fileName: "town_hall_recording.mp4", verdict: "deepfake", deepfakeScore: 0.94, createdAt: hoursAgo(8.9) },
];
