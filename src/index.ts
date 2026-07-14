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
  const fiveHours = 5 * 60 * 60 * 1000;

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
    if (closesAt - now > fiveHours) {
      console.log(`  [skip] "${e.title}" — closes in ${minsLeft} min (more than 3h)`);
      return false;
    }
    console.log(`  [pick] "${e.title}" — closes in ${minsLeft} min`);
    return true;
  });
}

// ─── AI Pick ──────────────────────────────────────────────────────────────────

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: "gemini-3.1-flash-lite",
  tools: [{ googleSearchRetrieval: {} }],
  generationConfig: {
    responseMimeType: "application/json",
    responseSchema: {
      type: SchemaType.OBJECT,
      properties: {
        reasoning: {
          type: SchemaType.STRING,
          description: "Step-by-step reasoning evaluating the probability of each option based on web search results.",
        },
        chosenOptionId: {
          type: SchemaType.STRING,
          description: "The exact ID of the option that has the highest probability of winning.",
        },
      },
      required: ["reasoning", "chosenOptionId"],
    },
  },
});

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function getAIPick(event: Event): Promise<string | null> {
  const optionsList = event.options
    .map((o) => `[${o.id}] ${o.optionText}`)
    .join("\n");

  const prompt = `You are analyzing a prediction market event. Based on the question and options, pick the single most likely outcome.
Search the web for any relevant real-world context, statistics, recent news, or event details to ground your prediction.

Question: ${event.title}
${event.description ? `Context: ${event.description}` : ""}

Options:
${optionsList}

Perform a logical step-by-step reasoning analysis first. Evaluate the probability/likelihood of each option winning. Then, output your selected option ID in the JSON response structure.`;

  const result = await model.generateContent(prompt);
  const responseText = result.response.text();

  try {
    const parsed = JSON.parse(responseText);
    console.log(`  AI Reasoning: ${parsed.reasoning}`);
    const chosenOptionId = parsed.chosenOptionId?.trim();
    const match = event.options.find((o) => o.id === chosenOptionId);
    if (!match) {
      console.warn(`  AI returned unrecognized option ID in JSON: "${chosenOptionId}"`);
      return null;
    }
    return match.id;
  } catch (err: any) {
    console.error("  Failed to parse AI response JSON:", err.message);
    console.debug("  Raw AI response was:", responseText);
    return null;
  }
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
  amount: number
): Promise<void> {
  await api.post("/v1/picks", {
    eventId,
    optionId,
    amount,
    confidence: 70,
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
      const optionId = await getAIPick(event);
      if (!optionId) {
        console.log(`  Skipped — AI could not determine an option`);
      } else {
        const amount = event.minBetAmount || 10;
        await placePick(event.id, optionId, amount);
        const optionText = event.options.find((o) => o.id === optionId)?.optionText;
        console.log(`  Picked: "${optionText}" (amount: ${amount}, confidence: 70%)`);
      }
    } catch (err: any) {
      const status = err?.status ?? err?.response?.status;
      if (status === 429) {
        console.warn(`  Skipped — Gemini rate limit hit, will retry next run`);
      } else {
        console.error(`  Error: ${err.response?.data?.message || err.message}`);
      }
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
