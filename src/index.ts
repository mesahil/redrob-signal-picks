import axios, { AxiosInstance } from "axios";
import { GoogleGenerativeAI } from "@google/generative-ai";

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
    const res = await api.get("/v1/events", {
      params: { status: "OPENED", limit: 50, ...(cursor ? { cursor } : {}) },
    });
    const payload = res.data?.data;
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
const model = genAI.getGenerativeModel({ model: "gemini-3.1-flash-lite" });

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function getAIPick(event: Event): Promise<string | null> {
  const optionsList = event.options
    .map((o) => `[${o.id}] ${o.optionText}`)
    .join("\n");

  const prompt = `You are analyzing a prediction market event. Based on the question and options, pick the single most likely outcome.

Question: ${event.title}
${event.description ? `Context: ${event.description}` : ""}

Options:
${optionsList}

Reply with ONLY the raw option ID of your chosen option — no brackets, no explanation, nothing else. Example format: d0c14d45-9cf1-470f-84bb-ea8bccc39a40`;

  const result = await model.generateContent(prompt);
  const text = result.response.text().trim().replace(/^\[|\]$/g, "");

  const match = event.options.find((o) => o.id === text);
  if (!match) {
    console.warn(`  AI returned unrecognized option ID: "${text}"`);
    return null;
  }
  return match.id;
}

// ─── Reveal Picks ────────────────────────────────────────────────────────────

interface Pick {
  id: string;
  resultViewed: boolean;
  status: string;
  event: { title: string };
}

async function fetchUnrevealedPicks(): Promise<Pick[]> {
  const allPicks: Pick[] = [];
  let cursor: string | null = null;

  do {
    const res = await api.get("/v1/picks", {
      params: {
        status: ["WON", "LOST"],
        limit: 50,
        ...(cursor ? { cursor } : {}),
      },
    });
    const payload = res.data?.data;
    const page: Pick[] = payload?.picks || [];
    allPicks.push(...page);
    cursor = payload?.pagination?.nextCursor ?? null;
  } while (cursor);

  return allPicks.filter((p) => !p.resultViewed);
}

async function revealPick(pickId: string): Promise<void> {
  await api.patch(`/v1/picks/${pickId}/result-viewed`);
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
