import { action } from "./_generated/server";
import { v } from "convex/values";
import { DeepgramClient } from "@deepgram/sdk";
import OpenAI from "openai";
import { buildConversationRewriteMessages } from "./prompt";

if (!process.env.DEEPGRAM_API_KEY) {
  console.warn(
    "DEEPGRAM_API_KEY is not set - set it in your Convex env vars or .env.local",
  );
}

if (!process.env.HUGGING_FACE_API_KEY) {
  console.warn(
    "HUGGING_FACE_API_KEY is not set - set it in your Convex env vars or .env.local",
  );
}

if (!process.env.OPENAI_API_KEY) {
  console.warn(
    "OPENAI_API_KEY is not set - conversational rewrite will use fallback mode",
  );
}

/**
 * Small helper: request with retry/backoff for 429s
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxTries = 3,
  baseDelayMs = 500,
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: any) {
      attempt++;
      const status = err?.status || err?.statusCode || err?.code;
      if (
        (status === 429 ||
          err?.message?.toLowerCase().includes("quota") ||
          err?.message?.toLowerCase().includes("rate limit")) &&
        attempt < maxTries
      ) {
        const wait = baseDelayMs * Math.pow(2, attempt - 1);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }

      const e = new Error(
        `Request failed${status ? ` (status: ${status})` : ""}: ${err?.message ?? err}`,
      );
      // @ts-ignore
      e.cause = err;
      throw e;
    }
  }
}

/**
 * Convert an ArrayBuffer or Uint8Array to base64 string
 */
function arrayBufferToBase64(buffer: ArrayBuffer | Uint8Array) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function concatUint8Arrays(chunks: Uint8Array[]) {
  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

/**
 * Map voice names to Deepgram Aura voices
 */
const voiceMap: Record<string, string> = {
  alloy: "aura-2-athena-en",
  echo: "aura-2-draco-en",
  fable: "aura-2-cora-en",
  nova: "aura-2-pluto-en",
  onyx: "aura-2-vesta-en",
  shimmer: "aura-2-orpheus-en",
};

const voicePairMap: Record<string, string> = {
  alloy: "echo",
  echo: "alloy",
  fable: "onyx",
  onyx: "fable",
  nova: "shimmer",
  shimmer: "nova",
};

type Speaker = "A" | "B";
type ConversationTurn = {
  speaker: Speaker;
  text: string;
};

const MAX_CONVERSATION_TURNS = 10;
const MAX_TTS_CONCURRENCY = 3;

function getCompanionVoice(voice: string) {
  return voicePairMap[voice] ?? "echo";
}

function cleanSpeechText(text: string) {
  const withoutLeadingLabel = text.replace(
    /^([A-Za-z][A-Za-z0-9 _().'-]{0,60})\s*:\s*/i,
    "",
  );
  const normalized = withoutLeadingLabel.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  if (/[.!?]$/.test(normalized)) return normalized;
  return `${normalized}.`;
}

function normalizeConversationTurns(turns: ConversationTurn[]) {
  const normalized: ConversationTurn[] = [];

  for (const turn of turns) {
    const cleaned = cleanSpeechText(turn.text);
    if (!cleaned) continue;

    const expectedSpeaker: Speaker =
      normalized.length % 2 === 0 ? "A" : "B";

    normalized.push({
      speaker: turn.speaker ?? expectedSpeaker,
      text: cleaned,
    });

    if (normalized.length >= MAX_CONVERSATION_TURNS) break;
  }

  if (normalized.length === 1) {
    normalized.push({
      speaker: "B",
      text: "That is a great point. Could you explain that a bit more?",
    });
  }

  // Enforce alternating speakers for better conversational pacing.
  for (let i = 1; i < normalized.length; i++) {
    if (normalized[i].speaker === normalized[i - 1].speaker) {
      normalized[i].speaker = normalized[i - 1].speaker === "A" ? "B" : "A";
    }
  }

  return normalized;
}

function chooseSpeakerForLabel(
  normalizedLabel: string,
  speakerMap: Map<string, Speaker>,
): Speaker {
  if (speakerMap.has(normalizedLabel)) {
    return speakerMap.get(normalizedLabel)!;
  }

  if (normalizedLabel.includes("host")) {
    speakerMap.set(normalizedLabel, "A");
    return "A";
  }

  if (normalizedLabel.includes("guest")) {
    speakerMap.set(normalizedLabel, "B");
    return "B";
  }

  const nextSpeaker: Speaker = speakerMap.size % 2 === 0 ? "A" : "B";
  speakerMap.set(normalizedLabel, nextSpeaker);
  return nextSpeaker;
}

function parseTurnsFromLabeledInput(input: string): ConversationTurn[] {
  const normalizedInput = input.trim().replace(/^\s*["']|["']\s*$/g, "");
  if (!normalizedInput) return [];

  // Capture "Label: content" blocks until the next "Label:" line.
  const blockRegex =
    /(?:^|\n)\s*([A-Za-z][A-Za-z0-9 _().'-]{0,60})\s*:\s*([\s\S]*?)(?=\n\s*[A-Za-z][A-Za-z0-9 _().'-]{0,60}\s*:|$)/g;

  const speakerMap = new Map<string, Speaker>();
  const parsedTurns: ConversationTurn[] = [];

  let match: RegExpExecArray | null;
  while ((match = blockRegex.exec(normalizedInput)) !== null) {
    const rawLabel = (match[1] ?? "").trim();
    const text = (match[2] ?? "").trim();
    if (!rawLabel || !text) continue;

    const normalizedLabel = rawLabel.toLowerCase().replace(/\s+/g, " ");
    const speaker = chooseSpeakerForLabel(normalizedLabel, speakerMap);

    parsedTurns.push({ speaker, text });
  }

  if (parsedTurns.length < 2) return [];
  return normalizeConversationTurns(parsedTurns);
}

function splitSentences(input: string) {
  const normalized = input.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  return (normalized.match(/[^.!?]+[.!?]?/g) ?? [])
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function createFallbackConversation(input: string): ConversationTurn[] {
  const sentences = splitSentences(input);
  if (sentences.length === 0) return [];

  const turns: ConversationTurn[] = [];
  let chunk = "";
  let chunkWords = 0;
  const maxWordsPerTurn = 28;

  for (const sentence of sentences) {
    const words = sentence.split(/\s+/).length;

    if (chunk && chunkWords + words > maxWordsPerTurn) {
      turns.push({
        speaker: turns.length % 2 === 0 ? "A" : "B",
        text: chunk,
      });
      chunk = sentence;
      chunkWords = words;
    } else {
      chunk = chunk ? `${chunk} ${sentence}` : sentence;
      chunkWords += words;
    }
  }

  if (chunk) {
    turns.push({
      speaker: turns.length % 2 === 0 ? "A" : "B",
      text: chunk,
    });
  }

  if (turns.length < 2) {
    turns.push({
      speaker: "B",
      text: "Interesting. What is the key takeaway for listeners?",
    });
  }

  return normalizeConversationTurns(turns);
}

function parseOpenAIConversation(raw: string): ConversationTurn[] {
  const cleaned = raw
    .trim()
    .replace(/^```json/i, "")
    .replace(/^```/, "")
    .replace(/```$/, "")
    .trim();

  const parsed = JSON.parse(cleaned) as {
    turns?: { speaker?: string; text?: string }[];
  };

  if (!parsed.turns || !Array.isArray(parsed.turns)) {
    return [];
  }

  const toSpeaker = (speaker: string | undefined, index: number): Speaker => {
    if (speaker?.toUpperCase() === "B") return "B";
    if (speaker?.toUpperCase() === "A") return "A";
    return index % 2 === 0 ? "A" : "B";
  };

  const turns: ConversationTurn[] = parsed.turns
    .map((turn, index) => ({
      speaker: toSpeaker(turn.speaker, index),
      text: turn.text ?? "",
    }))
    .filter((turn) => turn.text.trim().length > 0);

  return normalizeConversationTurns(turns);
}

async function rewriteToConversation(input: string): Promise<ConversationTurn[]> {
  if (!process.env.OPENAI_API_KEY) return [];

  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  try {
    const completion = await withRetry(
      () =>
        openai.chat.completions.create({
          model: "gpt-4o-mini",
          temperature: 0.8,
          response_format: { type: "json_object" },
          messages: buildConversationRewriteMessages(input),
        }),
      2,
      400,
    );

    const content = completion.choices[0]?.message?.content ?? "";
    if (!content) return [];
    return parseOpenAIConversation(content);
  } catch (error) {
    console.warn("Conversation rewrite failed, using fallback mode:", error);
    return [];
  }
}

async function buildConversationTurns(input: string): Promise<ConversationTurn[]> {
  const labeledTurns = parseTurnsFromLabeledInput(input);
  if (labeledTurns.length >= 2) return labeledTurns;

  const rewrittenTurns = await rewriteToConversation(input);
  if (rewrittenTurns.length >= 2) return rewrittenTurns;

  return createFallbackConversation(input);
}

async function generateConversationalDeepgramTTS(
  input: string,
  voiceA: string,
  voiceB: string,
): Promise<Uint8Array> {
  const turns = await buildConversationTurns(input);

  if (turns.length === 0) {
    return await withRetry(() => generateDeepgramTTS(input, voiceA));
  }

  const chunks = await generateTurnAudiosWithConcurrency(turns, voiceA, voiceB);

  return concatUint8Arrays(chunks);
}

async function generateTurnAudiosWithConcurrency(
  turns: ConversationTurn[],
  voiceA: string,
  voiceB: string,
): Promise<Uint8Array[]> {
  const results = new Array<Uint8Array>(turns.length);
  const workerCount = Math.min(MAX_TTS_CONCURRENCY, turns.length);
  let nextIndex = 0;

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (true) {
        const currentIndex = nextIndex;
        nextIndex += 1;
        if (currentIndex >= turns.length) break;

        const turn = turns[currentIndex];
        const voice = turn.speaker === "A" ? voiceA : voiceB;

        try {
          results[currentIndex] = await withRetry(() =>
            generateDeepgramTTS(turn.text, voice),
          );
        } catch (error: any) {
          throw new Error(
            `Failed to synthesize turn ${currentIndex + 1}: ${error?.message ?? error}`,
          );
        }
      }
    }),
  );

  return results;
}

export const generateAudioAction = action({
  args: { input: v.string(), voice: v.string(), voiceB: v.optional(v.string()) },
  handler: async (_, { voice, input, voiceB }) => {
    const normalizedInput = input.trim();
    if (!normalizedInput) {
      throw new Error("Input text is empty. Please provide text to generate audio.");
    }

    if (!process.env.DEEPGRAM_API_KEY) {
      throw new Error(
        "DEEPGRAM_API_KEY missing. Add it to your Convex environment variables.",
      );
    }

    const secondaryVoice =
      voiceB && voiceB !== voice ? voiceB : getCompanionVoice(voice);

    const audioContent = await generateConversationalDeepgramTTS(
      normalizedInput,
      voice,
      secondaryVoice,
    );

    return { base64: arrayBufferToBase64(audioContent), mime: "audio/mpeg" };
  },
});

/**
 * Generate audio using Deepgram TTS
 * uses aura-2-thalia-en or mapped voices
 */
async function generateDeepgramTTS(
  text: string,
  voice: string,
): Promise<Uint8Array> {
  const deepgram = new DeepgramClient();

  const modelId = voiceMap[voice] || "aura-2-orpheus-en";

  const response = await deepgram.speak.v1.audio.generate({
    text,
    model: modelId,
    encoding: "mp3",
  });

  const stream = response.stream();
  if (!stream) {
    throw new Error("Deepgram failed to return a stream.");
  }

  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  return concatUint8Arrays(chunks);
}

export const generateThumbnailAction = action({
  args: { prompt: v.string() },
  handler: async (_, { prompt }) => {
    if (!process.env.HUGGING_FACE_API_KEY) {
      throw new Error(
        "HUGGING_FACE_API_KEY missing. Add it to your Convex environment variables.",
      );
    }

    const result = await withRetry(async () => {
      const imageBuffer = await generateHuggingFaceImage(prompt);
      const base64 = arrayBufferToBase64(imageBuffer);
      return { base64, mime: "image/png" };
    });

    return result;
  },
});

/**
 * Generate image using Hugging Face Inference API (Stable Diffusion)
 */
async function generateHuggingFaceImage(prompt: string): Promise<Uint8Array> {
  const apiKey = process.env.HUGGING_FACE_API_KEY;
  const modelId = "stabilityai/stable-diffusion-2-1";

  const response = await fetch(
    `https://api-inference.huggingface.co/models/${modelId}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey!}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: prompt }),
    },
  );

  if (!response.ok) {
    throw new Error(`Hugging Face API error: ${response.statusText}`);
  }

  return new Uint8Array(await response.arrayBuffer());
}
