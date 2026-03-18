import { action } from "./_generated/server";
import { v } from "convex/values";
import { DeepgramClient } from "@deepgram/sdk";

if (!process.env.DEEPGRAM_API_KEY) {
  console.warn(
    "DEEPGRAM_API_KEY is not set — set it in your Convex env vars or .env.local",
  );
}

if (!process.env.HUGGING_FACE_API_KEY) {
  console.warn(
    "HUGGING_FACE_API_KEY is not set — set it in your Convex env vars or .env.local",
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
      // If rate limit / quota (429) and we have attempts left, retry with backoff
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
      // rethrow enriched error
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

export const generateAudioAction = action({
  args: { input: v.string(), voice: v.string() },
  handler: async (_, { voice, input }) => {
    if (!process.env.DEEPGRAM_API_KEY) {
      throw new Error(
        "DEEPGRAM_API_KEY missing. Add it to your Convex environment variables.",
      );
    }

    const result = await withRetry(async () => {
      const audioContent = await generateDeepgramTTS(input, voice);
      const base64 = arrayBufferToBase64(audioContent);
      return { base64, mime: "audio/mpeg" };
    });

    return result;
  },
});

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
  });

  const stream = response.stream();
  if (!stream) {
    throw new Error("Deepgram failed to return a stream.");
  }

  // Convert the Web ReadableStream to a Buffer
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
