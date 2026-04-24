import type OpenAI from "openai";

const CONVERSATION_REWRITE_SYSTEM_PROMPT = `
You convert user text into a 2-speaker podcast dialogue.

Rules:
1. Output strict JSON only with shape: {"turns":[{"speaker":"A|B","text":"..."}]}
2. Keep 6-12 turns.
3. Keep each turn to one coherent spoken thought.
4. Keep wording natural and expressive for spoken audio.
5. Never include labels in text like "Host:", "Guest:", "Alex:", "(Maya)".
6. Do not merge two speakers into one turn.
7. Preserve technical meaning and important facts from the input.
8. Use punctuation for speaking rhythm, but avoid emoji.
`.trim();

type ExamplePair = {
  input: string;
  output: { turns: { speaker: "A" | "B"; text: string }[] };
};

const FEW_SHOT_EXAMPLES: ExamplePair[] = [
  {
    input:
      "Create a short conversation on why serverless architecture helps ML inference.",
    output: {
      turns: [
        {
          speaker: "A",
          text: "Today we are unpacking serverless inference. Why is it getting so much traction right now?",
        },
        {
          speaker: "B",
          text: "Because teams can scale automatically with spiky demand, without keeping idle GPU or CPU capacity online.",
        },
        {
          speaker: "A",
          text: "So cost efficiency is a core benefit. What about deployment speed?",
        },
        {
          speaker: "B",
          text: "It is faster too. You package focused functions, connect event triggers, and ship updates with less operational overhead.",
        },
        {
          speaker: "A",
          text: "And the tradeoff?",
        },
        {
          speaker: "B",
          text: "Cold starts and tighter execution limits, so architecture decisions still matter.",
        },
      ],
    },
  },
  {
    input:
      "Host (Alex): We are discussing AI observability. Guest (Maya): Logs, traces, and evals are essential. Host (Alex): Why not just monitor latency? Guest (Maya): Because latency alone misses quality regressions.",
    output: {
      turns: [
        {
          speaker: "A",
          text: "We are discussing AI observability today. Is latency monitoring alone enough?",
        },
        {
          speaker: "B",
          text: "Not even close. You also need logs, traces, and evaluation signals to detect quality drift.",
        },
        {
          speaker: "A",
          text: "So a fast response can still be wrong?",
        },
        {
          speaker: "B",
          text: "Exactly. Latency tells you speed, but observability tells you whether the output is still trustworthy.",
        },
      ],
    },
  },
];

export function buildConversationRewriteMessages(
  input: string,
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: CONVERSATION_REWRITE_SYSTEM_PROMPT,
    },
  ];

  for (const example of FEW_SHOT_EXAMPLES) {
    messages.push({
      role: "user",
      content: example.input,
    });
    messages.push({
      role: "assistant",
      content: JSON.stringify(example.output),
    });
  }

  messages.push({
    role: "user",
    content: input.slice(0, 5000),
  });

  return messages;
}

