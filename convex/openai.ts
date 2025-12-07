import { action } from "./_generated/server";
import { v } from "convex/values";

import OpenAI from "openai";
import { SpeechCreateParams } from "openai/resources/audio/speech.mjs";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

if (!process.env.OPENAI_API_KEY) {
  console.warn("OPENAI_API_KEY is not set — set it in your Convex env vars or .env.local");
}

/**
 * Small helper: request with retry/backoff for 429s
 */
async function withRetry<T>(fn: () => Promise<T>, maxTries = 3, baseDelayMs = 500): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err: any) {
      attempt++;
      const status = err?.status || err?.statusCode || err?.code;
      // If rate limit / quota (429) and we have attempts left, retry with backoff
      if ((status === 429 || err?.message?.toLowerCase().includes("quota") || err?.message?.toLowerCase().includes("rate limit")) && attempt < maxTries) {
        const wait = baseDelayMs * Math.pow(2, attempt - 1);
        await new Promise((r) => setTimeout(r, wait));
        continue;
      }
      // rethrow enriched error
      const e = new Error(`OpenAI request failed${status ? ` (status: ${status})` : ""}: ${err?.message ?? err}`);
      // @ts-ignore
      e.cause = err;
      throw e;
    }
  }
}

/**
 * Convert an ArrayBuffer or Buffer-like to base64 string
 */
function arrayBufferToBase64(ab: ArrayBuffer | Buffer) {
  const buf = Buffer.isBuffer(ab) ? ab : Buffer.from(ab);
  return buf.toString("base64");
}

export const generateAudioAction = action({
  args: { input: v.string(), voice: v.string() },
  handler: async (_, { voice, input }) => {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY missing. Add it to your Convex environment variables.");
    }

    const result = await withRetry(async () => {
      // create audio
      const mp3 = await openai.audio.speech.create({
        model: "tts-1",
        voice: voice as SpeechCreateParams["voice"],
        input,
      } as any);

      // SDK may return a Node Response-like object with arrayBuffer()
      // or a stream - attempt arrayBuffer first.
      let ab: ArrayBuffer | Buffer;
      if (typeof (mp3 as any).arrayBuffer === "function") {
        ab = await (mp3 as any).arrayBuffer();
      } else if (mp3 instanceof ArrayBuffer) {
        ab = mp3;
      } else if (Buffer.isBuffer(mp3)) {
        ab = mp3;
      } else {
        // Fallback: try to read as any
        throw new Error("Unknown audio response shape from OpenAI SDK.");
      }

      const base64 = arrayBufferToBase64(ab);
      // Return base64 with mime — client will convert to blob
      return { base64, mime: "audio/mpeg" };
    });

    return result; // { base64, mime }
  },
});

export const generateThumbnailAction = action({
  args: { prompt: v.string() },
  handler: async (_, { prompt }) => {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY missing. Add it to your Convex environment variables.");
    }

    const result = await withRetry(async () => {
      const response = await openai.images.generate({
        model: "dall-e-3",
        prompt,
        size: "1024x1024",
        quality: "standard",
        n: 1,
      } as any);

      const first = response.data?.[0];
      // Newer SDK often returns b64_json. Older flows return a url to fetch.
      if (first?.b64_json) {
        return { base64: first.b64_json, mime: "image/png" };
      }

      const url = first?.url;
      if (!url) {
        throw new Error("No image URL or b64_json returned by OpenAI images.generate");
      }

      // fetch the URL and convert to base64
      const imageRes = await fetch(url);
      if (!imageRes.ok) {
        throw new Error(`Failed to fetch generated image: ${imageRes.status} ${imageRes.statusText}`);
      }
      const contentType = imageRes.headers.get("content-type") || "image/png";
      const arrayBuffer = await imageRes.arrayBuffer();
      const base64 = arrayBufferToBase64(arrayBuffer);
      return { base64, mime: contentType };
    });

    return result; // { base64, mime }
  },
});
