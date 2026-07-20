import axios, { AxiosInstance } from "axios";
import { GoogleGenerativeAI, SchemaType } from "@google/generative-ai";

// ─── Config ──────────────────────────────────────────────────────────────────

const API_BASE_URL =
  process.env.API_BASE_URL || "https://api-signal.redrob.io/api";

let accessToken = process.env.ACCESS_TOKEN || "";
let refreshToken = process.env.REFRESH_TOKEN || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";

if (!accessToken || !refreshToken || !GEMINI_API_KEY) {
  console.error("Missing required env vars: ACCESS_TOKEN, REFRESH_TOKEN, GEMINI_API_KEY");
  process.exit(1);
}

// ─── API Client ───────────────────────────────────────────────────────────────

const api: AxiosInstance = axios.create({ baseURL: API_BASE_URL });

api.interceptors.request.use((config) => {
  config.headers.Authorization = `Bearer ${accessToken}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    if (error.response?.status === 401 && !error.config._retried) {
      error.config._retried = true;
      const refreshed = await refreshAccessToken();
      if (refreshed) {
        error.config.headers.Authorization = `Bearer ${accessToken}`;
        return api.request(error.config);
      }
    }
    return Promise.reject(error);
  }
);

async function refreshAccessToken(): Promise<boolean> {
  try {
    const res = await axios.post(
      `${API_BASE_URL}/v1/auth/refresh-token`,
      { refresh_token: refreshToken },
    );
    const data = res.data?.data;
    accessToken = data?.access_token || data?.data?.access_token;
    refreshToken = data?.refresh_token || data?.data?.refresh_token;
    console.log("Token refreshed successfully");
    return true;
  } catch (err: any) {
    console.error("Token refresh failed:", err.message);
    return false;
  }
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface EventOption {
  id: string;
  optionText: string;
}

interface Event {
  id: string;
  title: string;
  description?: string | null;
  status: string;
  minBetAmount?: number;
  closesAt?: string | null;
  options: EventOption[];
  userPick?: unknown | null;
}

// ─── Fetch Events ─────────────────────────────────────────────────────────────

async function fetchOpenEvents(): Promise<Event[]> {
  const allEvents: Event[] = [];
  let cursor: string | null = null;

  do {
    const res: any = await api.get("/v1/events", {
      params: { status: "OPENED", limit: 50, ...(cursor ? { cursor } : {}) },
    });
    const payload: any = res.data?.data;
    const page: Event[] = payload?.events || [];
    allEvents.push(...page);
    cursor = payload?.pagination?.nextCursor ?? null;
  } while (cursor);

  console.log(`Total events fetched: ${allEvents.length}`);

  const now = Date.now();
  const fifteenHours = 15 * 60 * 60 * 1000;

  return allEvents.filter((e) => {
    if (e.userPick) {
      console.log(`  [skip] "${e.title}" — already picked`);
      return false;
    }
    if (!e.closesAt) {
      console.log(`  [skip] "${e.title}" — no closesAt`);
      return false;
    }
    const closesAt = new Date(e.closesAt).getTime();
    const minsLeft = Math.round((closesAt - now) / 60000);
    if (closesAt <= now) {
      console.log(`  [skip] "${e.title}" — already closed (closesAt: ${e.closesAt})`);
      return false;
    }
    if (closesAt - now > fifteenHours) {
      console.log(`  [skip] "${e.title}" — closes in ${minsLeft} min (more than 15h)`);
      return false;
    }
    console.log(`  [pick] "${e.title}" — closes in ${minsLeft} min`);
    return true;
  });
}

// ─── AI Pick Setup ────────────────────────────────────────────────────────────

// Parse multiple API keys (comma-separated GEMINI_API_KEYS or single GEMINI_API_KEY)
const apiKeys: string[] = process.env.GEMINI_API_KEYS
  ? process.env.GEMINI_API_KEYS.split(",").map((k) => k.trim()).filter(Boolean)
  : process.env.GEMINI_API_KEY
    ? [process.env.GEMINI_API_KEY.trim()]
    : [];

if (apiKeys.length === 0) {
  console.error("Missing required env vars: GEMINI_API_KEYS or GEMINI_API_KEY");
  process.exit(1);
}

// Model Priority List requested
const MODEL_PRIORITY_LIST: string[] = [
  // "gemini-2.5-flash",
  "gemma-4-31b-it",
  "gemini-3.1-flash-lite",
];

// Cache GoogleGenerativeAI instances per API Key
const genAIInstances = new Map<string, GoogleGenerativeAI>();
function getGenAIInstance(key: string): GoogleGenerativeAI {
  if (!genAIInstances.has(key)) {
    genAIInstances.set(key, new GoogleGenerativeAI(key));
  }
  return genAIInstances.get(key)!;
}

// Sticky pointers: current Key index and current Model index
let currentKeyIndex = 0;
let currentModelIndex = 0;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

interface AIPickResult {
  optionId: string;
  confidence: number;
  modelUsed: string;
}

async function getAIPick(event: Event): Promise<AIPickResult> {
  const optionsList = event.options
    .map((o) => `Option ID: "${o.id}" -> ${o.optionText}`)
    .join("\n");

  const prompt = `You are a prediction market analyst. You MUST evaluate this event and select the single option with the highest logical and statistical probability of winning.

Question: ${event.title}
${event.description ? `Context: ${event.description}` : ""}
${event.closesAt ? `Closes At: ${event.closesAt}` : ""}

Options:
${optionsList}

Follow this exact evaluation process:
1. Determine the category (Sports, News/Politics, Crypto/Finance, Entertainment, or Miscellaneous).
2. Perform a multi-perspective analysis for EACH option:
   - What must occur for this option to win?
   - What risks could cause this option to lose?
3. Apply statistical & domain rules:
   - For binary Yes/No: Assess default base-rate (No is often default unless active triggers occur).
   - For sports/competitions: Favor higher base-probability outcomes unless strong contradictory factors exist.
   - For price/time targets: Assess whether the target is realistic within the timeframe before closesAt.
4. Estimate an approximate percentage probability for every option.
5. Choose the option ID with the single highest probability.
6. Provide your evaluated confidence score (integer between 50 and 95) for your choice.`;

  // Cascade across API Keys
  while (currentKeyIndex < apiKeys.length) {
    const activeApiKey = apiKeys[currentKeyIndex];
    const aiInstance = getGenAIInstance(activeApiKey);

    // Cascade across Models for the active API Key
    while (currentModelIndex < MODEL_PRIORITY_LIST.length) {
      const activeModelName = MODEL_PRIORITY_LIST[currentModelIndex];

      try {
        console.log(
          `  [AI] Querying Key #${currentKeyIndex + 1} (${activeApiKey.slice(0, 6)}...) with model "${activeModelName}"`
        );

        const modelConfig: any = {
          model: activeModelName,
          generationConfig: {
            responseMimeType: "application/json",
            responseSchema: {
              type: SchemaType.OBJECT,
              properties: {
                reasoning: {
                  type: SchemaType.STRING,
                  description: "Step-by-step comparative evaluation of each option's probability.",
                },
                chosenOptionId: {
                  type: SchemaType.STRING,
                  description: "The exact ID of the option with the highest winning probability.",
                },
                confidence: {
                  type: SchemaType.NUMBER,
                  description: "Confidence score percentage (50 to 95) for the chosen option.",
                },
              },
              required: ["reasoning", "chosenOptionId", "confidence"],
            },
          },
        };

        const enableInternet = activeModelName !== "gemini-3.1-flash-lite";
        let result: any;

        if (enableInternet) {
          try {
            const modelWithTools = aiInstance.getGenerativeModel({
              ...modelConfig,
              tools: [{ googleSearch: {} } as any],
            });
            result = await modelWithTools.generateContent(prompt);
          } catch (toolErr: any) {
            console.warn(
              `  [Internet Notice] "${activeModelName}" search tools failed (${toolErr.message.slice(0, 70)}...). Falling back to basic mode...`
            );
            const basicModel = aiInstance.getGenerativeModel(modelConfig);
            result = await basicModel.generateContent(prompt);
          }
        } else {
          // gemini-3.1-flash-lite runs directly in basic mode without internet search tools
          const basicModel = aiInstance.getGenerativeModel(modelConfig);
          result = await basicModel.generateContent(prompt);
        }

        const responseText = result.response.text();
        const parsed = JSON.parse(responseText);

        console.log(`  AI Reasoning (${activeModelName}): ${parsed.reasoning}`);
        const chosenOptionId = parsed.chosenOptionId?.trim();
        const match = event.options.find((o) => o.id === chosenOptionId);

        const parsedConfidence = Math.min(
          95,
          Math.max(50, Math.round(Number(parsed.confidence) || 70))
        );

        if (match) {
          return { optionId: match.id, confidence: parsedConfidence, modelUsed: activeModelName };
        }

        console.warn(`  AI returned unrecognized option ID: "${chosenOptionId}"`);
      } catch (err: any) {
        console.warn(
          `  [Model Failover] Key #${currentKeyIndex + 1} + "${activeModelName}" failed: ${err.message}`
        );

        currentModelIndex++;
        if (currentModelIndex < MODEL_PRIORITY_LIST.length) {
          console.log(
            `  [Next Model] Advancing to model "${MODEL_PRIORITY_LIST[currentModelIndex]}" for Key #${currentKeyIndex + 1}...`
          );
          await sleep(1000);
        }
      }
    }

    // All models on current API Key exhausted -> switch to next API Key and reset model index to 0
    currentKeyIndex++;
    currentModelIndex = 0;

    if (currentKeyIndex < apiKeys.length) {
      console.log(
        `  [Key Switch] All models exhausted for Key #${currentKeyIndex}. Switching to API Key #${currentKeyIndex + 1}!`
      );
      await sleep(2000);
    } else {
      console.error("  [All Keys Exhausted] All API keys and models exhausted.");
    }
  }

  // Fallback if all keys & all models fail (Guarantees 100% event pick coverage, no event skipped)
  console.warn(`  Fallback triggered: Picking first option "${event.options[0]?.optionText}"`);
  return { optionId: event.options[0].id, confidence: 50, modelUsed: "fallback-default" };
}

// ─── Reveal Picks ────────────────────────────────────────────────────────────

interface Pick {
  id: string;
  resultViewed: boolean;
  status: string;
  event: { title: string };
}

async function fetchUnrevealedPicks(): Promise<Pick[]> {
  const unrevealed: Pick[] = [];
  let cursor: string | null = null;
  let totalFetched = 0;
  let consecutiveViewed = 0;
  const MAX_PICKS = 200;
  const STOP_AFTER_VIEWED = 4;

  do {
    const res: any = await api.get("/v1/picks", {
      params: {
        status: ["WON", "LOST"],
        sortBy: "revealFirst",
        limit: 50,
        ...(cursor ? { cursor } : {}),
      },
      paramsSerializer: { indexes: null },
    });

    const payload: any = res.data?.data;
    const picks: Pick[] = payload?.picks || [];

    for (const pick of picks) {
      totalFetched++;
      if (!pick.resultViewed) {
        consecutiveViewed = 0;
        unrevealed.push(pick);
        console.log(`  [reveal] Queued: "${pick.event?.title}" — ${pick.status}`);
      } else {
        consecutiveViewed++;
        if (consecutiveViewed >= STOP_AFTER_VIEWED) {
          console.log(`  [reveal] Hit ${STOP_AFTER_VIEWED} consecutive viewed picks — stopping early at ${totalFetched} total`);
          return unrevealed;
        }
      }

      if (totalFetched >= MAX_PICKS) {
        console.log(`  [reveal] Reached ${MAX_PICKS} pick limit — stopping`);
        return unrevealed;
      }
    }

    cursor = payload?.pagination?.nextCursor ?? null;
  } while (cursor);

  console.log(`  [reveal] Done — fetched ${totalFetched}, unrevealed: ${unrevealed.length}`);
  return unrevealed;
}

async function revealPick(pickId: string): Promise<void> {
  console.log(`  [reveal] PATCH /v1/picks/${pickId}/result-viewed`);
  const res = await api.patch(`/v1/picks/${pickId}/result-viewed`);
  console.log(`  [reveal] Response status: ${res.status}`);
}

// ─── Place Pick ───────────────────────────────────────────────────────────────

async function placePick(
  eventId: string,
  optionId: string,
  amount: number,
  confidence: number = 70
): Promise<void> {
  await api.post("/v1/picks", {
    eventId,
    optionId,
    amount,
    confidence,
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\n[${new Date().toISOString()}] Starting auto-pick run...`);

  let events: Event[];
  try {
    events = await fetchOpenEvents();
  } catch (err: any) {
    console.error("Failed to fetch events:", err.message);
    process.exit(1);
  }

  console.log(`Found ${events.length} open event(s) without a pick`);

  for (const event of events) {
    console.log(`\nProcessing: "${event.title}"`);
    try {
      const { optionId, confidence, modelUsed } = await getAIPick(event);
      const amount = event.minBetAmount || 10;
      await placePick(event.id, optionId, amount, confidence);
      const optionText = event.options.find((o) => o.id === optionId)?.optionText;
      console.log(
        `  [PICKED via ${modelUsed}] "${optionText}" (amount: ${amount}, confidence: ${confidence}%)`
      );
    } catch (err: any) {
      console.error(`  Error processing event "${event.title}":`, err.response?.data?.message || err.message);
    }
    await sleep(4000); // stay under 15 req/min free tier limit
  }

  // ── Reveal settled picks ──────────────────────────────────────────────────
  console.log("\nChecking for unrevealed results...");
  try {
    const unrevealed = await fetchUnrevealedPicks();
    console.log(`Found ${unrevealed.length} unrevealed pick(s)`);
    for (const pick of unrevealed) {
      try {
        await revealPick(pick.id);
        console.log(`  Revealed: "${pick.event.title}" — ${pick.status}`);
      } catch (err: any) {
        console.error(`  Error revealing pick ${pick.id}: ${err.response?.data?.message || err.message}`);
      }
    }
  } catch (err: any) {
    console.error("Failed to fetch unrevealed picks:", err.message);
  }

  console.log(`\n[${new Date().toISOString()}] Run complete.`);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
