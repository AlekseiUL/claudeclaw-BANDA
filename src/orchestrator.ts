import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

import { runAgent, UsageInfo } from './agent.js';
import { loadAgentConfig, listAgentIds, resolveAgentClaudeMd } from './agent-config.js';
import { PROJECT_ROOT, AGENT_ID, TELEGRAM_BOT_TOKEN, ALLOWED_CHAT_ID, DASHBOARD_TOKEN } from './config.js';
import { logToHiveMind, createInterAgentTask, updateInterAgentTaskStatus, completeInterAgentTask, getScheduledChains, updateChainRun, getSession, setSession } from './db.js';
import type { ChainStep } from './db.js';
import { logger } from './logger.js';
import { boardAddTask, boardCompleteTask, boardFailTask } from './task-board.js';

// ── Remote delegation (agent's own dashboard API) ────────────────────

interface RemoteDelegationResult {
  text: string | null;
  usage: UsageInfo | null;
}

/**
 * Try to delegate a task to an agent's own dashboard API.
 * Returns result if successful, null if agent has no dashboard port.
 * Throws on connection/HTTP errors (caller falls back to in-process).
 */
async function tryRemoteDelegation(
  agentId: string,
  prompt: string,
  port: number,
  timeoutMs: number,
  fromAgent: string,
): Promise<RemoteDelegationResult> {
  const url = `http://localhost:${port}/api/delegate?token=${DASHBOARD_TOKEN}`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agentId, prompt, timeoutMs, fromAgent }),
    signal: AbortSignal.timeout(timeoutMs + 10000),
  });

  if (!resp.ok) {
    throw new Error(`Agent dashboard returned ${resp.status}`);
  }

  const data = await resp.json() as {
    ok: boolean;
    result?: { text: string | null; usage?: UsageInfo };
    error?: string;
  };
  if (!data.ok) {
    throw new Error(data.error || 'Remote delegation returned ok=false');
  }

  return {
    text: data.result?.text ?? null,
    usage: (data.result?.usage as UsageInfo) ?? null,
  };
}

// ── Agent-to-user messaging ──────────────────────────────────────────

/**
 * Send a Telegram message FROM an agent's own bot to a chat.
 * Used so that the user sees messages from @TucoSalamanka_bot etc.
 * Fire-and-forget: never throws, logs errors silently.
 */
async function sendAsAgent(botToken: string, chatId: string, text: string): Promise<void> {
  try {
    await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: chatId, text }),
    });
  } catch (err) {
    logger.warn({ err, chatId }, 'Failed to send agent message to Telegram');
  }
}

// ── Types ────────────────────────────────────────────────────────────

export interface DelegationResult {
  agentId: string;
  text: string | null;
  usage: UsageInfo | null;
  taskId: string;
  durationMs: number;
}

export interface AgentInfo {
  id: string;
  name: string;
  description: string;
}

// ── Registry ─────────────────────────────────────────────────────────

/** Cache of available agents loaded at startup. */
let agentRegistry: AgentInfo[] = [];

/** Default timeout for a delegated task (10 minutes). */
const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000;

/**
 * Initialize the orchestrator by scanning `agents/` for valid configs.
 * Safe to call even if no agents are configured — the registry will be empty.
 */
export function initOrchestrator(): void {
  const ids = listAgentIds();
  agentRegistry = [];

  for (const id of ids) {
    try {
      const config = loadAgentConfig(id);
      agentRegistry.push({
        id,
        name: config.name,
        description: config.description,
      });
    } catch (err) {
      // Agent config is broken (e.g. missing token) — skip it but warn
      logger.warn({ agentId: id, err }, 'Skipping agent — config load failed');
    }
  }

  logger.info(
    { agents: agentRegistry.map((a) => a.id) },
    'Orchestrator initialized',
  );
}

/** Return all agents that were successfully loaded. */
export function getAvailableAgents(): AgentInfo[] {
  return [...agentRegistry];
}

// ── Health Check ─────────────────────────────────────────────────────

export interface AgentHealthStatus {
  id: string;
  name: string;
  botAlive: boolean;
  processAlive: boolean;
  error?: string;
}

/**
 * Check if an agent's Telegram bot is responsive (getMe API call).
 * Also checks if the launchd/systemd process is running.
 */
export async function checkAgentHealth(agentId: string): Promise<AgentHealthStatus> {
  const agent = agentRegistry.find((a) => a.id === agentId);
  if (!agent) {
    return { id: agentId, name: agentId, botAlive: false, processAlive: false, error: 'Агент не найден в реестре' };
  }

  let botAlive = false;
  let processAlive = false;
  let error: string | undefined;

  // Check Telegram bot via getMe
  try {
    const config = loadAgentConfig(agentId);
    const resp = await fetch(`https://api.telegram.org/bot${config.botToken}/getMe`, { signal: AbortSignal.timeout(5000) });
    botAlive = resp.ok;
    if (!resp.ok) error = `Bot API: ${resp.status}`;
  } catch (err) {
    error = `Bot API: ${err instanceof Error ? err.message : String(err)}`;
  }

  // Check launchd process
  try {
    const { execSync } = await import('child_process');
    const output = execSync(`launchctl list com.claudeclaw.agent-${agentId} 2>/dev/null || true`, { encoding: 'utf-8' });
    processAlive = output.includes('PID') || /^\d+/.test(output.trim());
  } catch {
    // Not running or launchctl not available
  }

  return { id: agentId, name: agent.name, botAlive, processAlive, error };
}

/**
 * Check health of all registered agents.
 */
export async function checkAllAgentsHealth(): Promise<AgentHealthStatus[]> {
  return Promise.all(agentRegistry.map((a) => checkAgentHealth(a.id)));
}

// ── Delegation ───────────────────────────────────────────────────────

/**
 * Parse a user message for delegation syntax.
 *
 * Supported forms:
 *   @agentId: prompt text
 *   @agentId prompt text   (only if agentId is a known agent)
 *   /delegate agentId prompt text
 *
 * Returns `{ agentId, prompt }` or `null` if no delegation detected.
 */
export function parseDelegation(
  message: string,
): { agentId: string; prompt: string } | null {
  const isKnownAgent = (id: string) => agentRegistry.some((a) => a.id === id);

  // /delegate agentId prompt
  const cmdMatch = message.match(
    /^\/delegate\s+(\S+)\s+([\s\S]+)/i,
  );
  if (cmdMatch && isKnownAgent(cmdMatch[1])) {
    return { agentId: cmdMatch[1], prompt: cmdMatch[2].trim() };
  }

  // @agentId: prompt (only for known agents - prevents false positives on emails/mentions)
  const atMatch = message.match(
    /^@(\S+?):\s*([\s\S]+)/,
  );
  if (atMatch && isKnownAgent(atMatch[1])) {
    return { agentId: atMatch[1], prompt: atMatch[2].trim() };
  }

  // @agentId prompt (only for known agents to avoid false positives)
  const atMatchNoColon = message.match(
    /^@(\S+)\s+([\s\S]+)/,
  );
  if (atMatchNoColon && isKnownAgent(atMatchNoColon[1])) {
    return { agentId: atMatchNoColon[1], prompt: atMatchNoColon[2].trim() };
  }

  return null;
}

/**
 * Delegate a task to another agent. Runs the agent's Claude Code session
 * in-process (same Node.js process) with the target agent's cwd and
 * system prompt.
 *
 * The delegation is logged to both `inter_agent_tasks` and `hive_mind`.
 *
 * @param agentId    Target agent identifier (must exist in agents/)
 * @param prompt     The task to delegate
 * @param chatId     Telegram chat ID (for DB tracking)
 * @param fromAgent  The requesting agent's ID (usually 'main')
 * @param onProgress Optional callback for status updates
 * @param timeoutMs  Maximum execution time (default 5 min)
 */
export async function delegateToAgent(
  agentId: string,
  prompt: string,
  chatId: string,
  fromAgent: string,
  onProgress?: (msg: string) => void,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<DelegationResult> {
  const agent = agentRegistry.find((a) => a.id === agentId);
  if (!agent) {
    const available = agentRegistry.map((a) => a.id).join(', ') || '(none)';
    throw new Error(
      `Agent "${agentId}" not found. Available: ${available}`,
    );
  }

  const taskId = crypto.randomUUID();
  const start = Date.now();

  // Record the task
  createInterAgentTask(taskId, fromAgent, agentId, chatId, prompt);
  logToHiveMind(
    fromAgent,
    chatId,
    'delegate',
    `Delegated to ${agentId}: ${prompt.slice(0, 100)}`,
  );

  const taskPreview = prompt.length > 80 ? prompt.slice(0, 80) + '...' : prompt;

  // Mark task as accepted + board
  updateInterAgentTaskStatus(taskId, 'accepted');
  boardAddTask(agent.name, taskPreview, '⏳');
  onProgress?.(`👋 ${agent.name} принял задачу: ${taskPreview}`);

  // Send "task received" message FROM the agent's own bot to user
  const agentConfig = loadAgentConfig(agentId);
  if (agentConfig.botToken) {
    const senderName = fromAgent === 'main' ? 'Начо' : fromAgent;
    void sendAsAgent(
      agentConfig.botToken,
      chatId,
      `👋 Task received from ${senderName}: ${taskPreview}`,
    );
  }

  // Mark task as in_progress before execution
  updateInterAgentTaskStatus(taskId, 'in_progress');

  // ── Try remote delegation first (agent's own dashboard API) ──
  // Skip if: no dashboard port, or we ARE the target agent (avoid recursion)
  if (agentConfig.dashboardPort && AGENT_ID !== agentId) {
    try {
      logger.info({ agentId, port: agentConfig.dashboardPort }, 'Attempting remote delegation');
      const remoteResult = await tryRemoteDelegation(
        agentId, prompt, agentConfig.dashboardPort, timeoutMs, fromAgent,
      );
      const durationMs = Date.now() - start;
      completeInterAgentTask(taskId, 'completed', remoteResult.text);
      logToHiveMind(agentId, chatId, 'delegate_result',
        `Completed remote delegation from ${fromAgent}: ${(remoteResult.text ?? '').slice(0, 120)}`);
      boardCompleteTask(agent.name);
      onProgress?.(`✅ ${agent.name} завершил задачу (${Math.round(durationMs / 1000)}с)`);
      if (agentConfig.botToken) {
        const senderName = fromAgent === 'main' ? 'Начо' : fromAgent;
        void sendAsAgent(agentConfig.botToken, chatId,
          `✅ Задача выполнена! Отправляю результат → ${senderName}`);
      }
      return { agentId, text: remoteResult.text, usage: remoteResult.usage, taskId, durationMs };
    } catch (err) {
      logger.warn({ agentId, err: err instanceof Error ? err.message : String(err) },
        'Remote delegation failed, falling back to in-process');
    }
  }

  // ── Fallback: in-process execution ──

  // ── Intermediate progress updates (60s rule) ──
  // Notify boss every 60s that agent is still working. Prevents "silent work" anxiety.
  const PROGRESS_INTERVAL_MS = 60_000;
  let progressCount = 0;
  const progressTimer = setInterval(() => {
    progressCount++;
    const elapsed = Math.round((Date.now() - start) / 1000);
    if (progressCount === 1) {
      onProgress?.(`⏳ ${agent.name} работает... (${elapsed}с)`);
    } else if (progressCount === 2) {
      onProgress?.(`⏳ ${agent.name} всё ещё работает (${elapsed}с). Задача объёмная.`);
    } else {
      onProgress?.(`⏳ ${agent.name}: ${elapsed}с, продолжает работу`);
    }
  }, PROGRESS_INTERVAL_MS);

  try {
    // Load agent config to get its system prompt
    const claudeMdPath = resolveAgentClaudeMd(agentId);
    let systemPrompt = '';
    if (claudeMdPath) {
      try {
        systemPrompt = fs.readFileSync(claudeMdPath, 'utf-8');
      } catch {
        // No CLAUDE.md for this agent — that's fine
      }
    }

    // Build the delegated prompt with agent role context
    const fullPrompt = systemPrompt
      ? `[Agent role — follow these instructions]\n${systemPrompt}\n[End agent role]\n\n${prompt}`
      : prompt;

    // Create an AbortController with timeout
    const abortCtrl = new AbortController();
    const timer = setTimeout(() => abortCtrl.abort(), timeoutMs);

    // Use persistent session when agent runs in its own process (AGENT_ID === agentId)
    // This lets the agent remember delegated tasks when user talks to it via Telegram
    const isSelfProcess = AGENT_ID === agentId;
    const existingSessionId = isSelfProcess && chatId ? getSession(chatId, agentId) : undefined;

    try {
      const result = await runAgent(
        fullPrompt,
        existingSessionId, // resume session if agent's own process, fresh otherwise
        () => {}, // no typing indicator needed for sub-delegation
        undefined, // no progress callback for inner agent
        agentConfig.model, // use agent's model from agent.yaml (e.g. haiku for Tuco)
        abortCtrl,
      );

      clearTimeout(timer);
      clearInterval(progressTimer);

      // Save session so Telegram messages can resume it
      if (isSelfProcess && chatId && result.newSessionId) {
        setSession(chatId, result.newSessionId, agentId);
      }

      const durationMs = Date.now() - start;
      completeInterAgentTask(taskId, 'completed', result.text);
      logToHiveMind(
        agentId,
        chatId,
        'delegate_result',
        `Completed delegation from ${fromAgent}: ${(result.text ?? '').slice(0, 120)}`,
      );

      boardCompleteTask(agent.name);
      onProgress?.(
        `✅ ${agent.name} завершил задачу и передал результат (${Math.round(durationMs / 1000)}с)`,
      );

      // Send "task done" message FROM the agent's own bot to user
      if (agentConfig.botToken) {
        const senderName = fromAgent === 'main' ? 'Начо' : fromAgent;
        void sendAsAgent(
          agentConfig.botToken,
          chatId,
          `✅ Задача выполнена! Отправляю результат → ${senderName}`,
        );
      }

      return {
        agentId,
        text: result.text,
        usage: result.usage,
        taskId,
        durationMs,
      };
    } catch (innerErr) {
      clearTimeout(timer);
      clearInterval(progressTimer);
      throw innerErr;
    }
  } catch (err) {
    clearInterval(progressTimer);
    boardFailTask(agent.name);
    const durationMs = Date.now() - start;
    const errMsg = err instanceof Error ? err.message : String(err);
    completeInterAgentTask(taskId, 'failed', errMsg);
    logToHiveMind(
      agentId,
      chatId,
      'delegate_error',
      `Delegation from ${fromAgent} failed: ${errMsg.slice(0, 120)}`,
    );
    throw err;
  }
}

// ── Reverse Channel (agent → coordinator) ───────────────────────────

/**
 * Report a message FROM a sub-agent TO the coordinator (main agent).
 * The message is sent via main bot's Telegram to ALLOWED_CHAT_ID,
 * logged to hive_mind, and written to the team board.
 *
 * Use case: Lalo found a trend, Tuco spotted a bug, etc.
 */
export async function reportToCoordinator(
  fromAgentId: string,
  message: string,
  category: 'insight' | 'alert' | 'request' | 'result' = 'insight',
): Promise<void> {
  const chatId = ALLOWED_CHAT_ID;
  if (!chatId) {
    logger.warn('reportToCoordinator: no ALLOWED_CHAT_ID set');
    return;
  }

  const agent = agentRegistry.find((a) => a.id === fromAgentId);
  const agentName = agent?.name ?? fromAgentId;

  const CATEGORY_EMOJI: Record<string, string> = {
    insight: '💡',
    alert: '🚨',
    request: '🙋',
    result: '📊',
  };
  const emoji = CATEGORY_EMOJI[category] ?? '📨';

  // Log to hive_mind
  logToHiveMind(fromAgentId, chatId, `report_${category}`, message.slice(0, 200));

  // Send via main bot's Telegram
  const mainBotToken = TELEGRAM_BOT_TOKEN;
  if (mainBotToken) {
    const text = `${emoji} Report from ${agentName}:\n\n${message}`;
    await sendAsAgent(mainBotToken, chatId, text);
  }

  logger.info({ fromAgentId, category }, 'Agent reported to coordinator');
}

// ── Fan-out Delegation (parallel) ───────────────────────────────────

export interface FanOutTask {
  agentId: string;
  prompt: string;
}

export interface FanOutResult {
  agentId: string;
  text: string | null;
  usage: UsageInfo | null;
  taskId: string;
  durationMs: number;
  error?: string;
}

/** Maximum number of agents to run in parallel. */
const MAX_CONCURRENT = 3;

/**
 * Delegate tasks to multiple agents in parallel.
 * Respects MAX_CONCURRENT limit.
 * Returns all results (including failures) — never throws.
 */
export async function delegateToMultiple(
  tasks: FanOutTask[],
  chatId: string,
  fromAgent: string,
  onProgress?: (msg: string) => void,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<FanOutResult[]> {
  if (tasks.length === 0) return [];

  const agentNames = tasks.map((t) => {
    const agent = agentRegistry.find((a) => a.id === t.agentId);
    return agent?.name ?? t.agentId;
  });
  onProgress?.(`🔀 Параллельная делегация: ${agentNames.join(', ')}`);

  const results: FanOutResult[] = [];

  // Process in batches of MAX_CONCURRENT
  for (let i = 0; i < tasks.length; i += MAX_CONCURRENT) {
    const batch = tasks.slice(i, i + MAX_CONCURRENT);

    const batchResults = await Promise.allSettled(
      batch.map((task) =>
        delegateToAgent(task.agentId, task.prompt, chatId, fromAgent, onProgress, timeoutMs),
      ),
    );

    for (let j = 0; j < batchResults.length; j++) {
      const r = batchResults[j];
      const task = batch[j];
      if (r.status === 'fulfilled') {
        results.push(r.value);
      } else {
        results.push({
          agentId: task.agentId,
          text: null,
          usage: null,
          taskId: '',
          durationMs: 0,
          error: r.reason instanceof Error ? r.reason.message : String(r.reason),
        });
      }
    }
  }

  const succeeded = results.filter((r) => !r.error).length;
  const failed = results.filter((r) => r.error).length;
  onProgress?.(`🔀 Parallel delegation completed: ${succeeded} succeeded, ${failed} errors`);

  return results;
}

// ── Summarize Before Forward ─────────────────────

/** Max size of prev_result before we truncate for next agent. */
const SUMMARIZE_THRESHOLD = 4000; // ~4KB

/**
 * If result exceeds threshold, keep first and last portions with a note.
 * Prevents next agent from choking on huge context.
 * Not an LLM summarization — just smart truncation to preserve structure.
 */
function summarizeForForward(text: string, stepAgent: string): string {
  if (text.length <= SUMMARIZE_THRESHOLD) return text;

  const headSize = Math.floor(SUMMARIZE_THRESHOLD * 0.6);
  const tailSize = Math.floor(SUMMARIZE_THRESHOLD * 0.3);
  const head = text.slice(0, headSize);
  const tail = text.slice(-tailSize);
  const omitted = text.length - headSize - tailSize;

  return `${head}\n\n[... ${omitted} символов от ${stepAgent} опущено для краткости ...]\n\n${tail}`;
}

// ── Chain Runner ────────────────────────────────────────────────────

/**
 * Execute a scheduled chain: step1 → step2 → ... → stepN.
 * Each step's result is available to the next step via {{prev_result}}.
 * Large results are summarized before forwarding.
 * Reports final result to coordinator via Telegram.
 */
export async function runChain(
  chainId: string,
  chainName: string,
  steps: ChainStep[],
  chatId: string,
): Promise<string> {
  updateChainRun(chainId, 'running');
  logToHiveMind('main', chatId, 'chain_start', `Chain "${chainName}" started (${steps.length} steps)`);

  let prevResult = '';
  let fullLastResult = ''; // keep full result for final report

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    // Use summarized version for intermediate steps, full for last
    const resultForPrompt = i === steps.length - 1 ? prevResult : summarizeForForward(prevResult, steps[Math.max(0, i - 1)]?.agent_id ?? 'unknown');
    const prompt = step.prompt_template.replace(/\{\{prev_result\}\}/g, resultForPrompt);

    logger.info({ chainId, step: i + 1, agent: step.agent_id }, `Chain step ${i + 1}/${steps.length}`);

    // ── Retry logic: 1 retry per step (откат при кривом результате) ──
    const MAX_RETRIES = 1;
    let lastErr: Error | null = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        if (attempt > 0) {
          logger.warn({ chainId, step: i + 1, attempt: attempt + 1, agent: step.agent_id }, 'Retrying failed chain step');
          await reportToCoordinator(
            'main',
            `🔁 Chain "${chainName}": retry step ${i + 1} (${step.agent_id}), attempt ${attempt + 1}`,
            'alert',
          );
        }

        const result = await delegateToAgent(
          step.agent_id,
          prompt,
          chatId,
          'main',
          undefined,
          DEFAULT_TIMEOUT_MS,
        );
        prevResult = result.text ?? '';
        fullLastResult = prevResult;
        lastErr = null;

        // Log if summarization will be applied for next step
        if (prevResult.length > SUMMARIZE_THRESHOLD && i < steps.length - 1) {
          logger.info(
            { chainId, step: i + 1, originalSize: prevResult.length, threshold: SUMMARIZE_THRESHOLD },
            'Result will be summarized before forwarding to next agent',
          );
        }
        break; // success — exit retry loop
      } catch (err) {
        lastErr = err instanceof Error ? err : new Error(String(err));
        if (attempt < MAX_RETRIES) {
          logger.warn({ chainId, step: i + 1, err: lastErr.message }, 'Chain step failed, will retry');
        }
      }
    }

    // If all retries exhausted — fail the chain
    if (lastErr) {
      const errMsg = lastErr.message;
      updateChainRun(chainId, 'failed', `Step ${i + 1} (${step.agent_id}): ${errMsg} (after ${MAX_RETRIES + 1} attempts)`);
      logToHiveMind('main', chatId, 'chain_failed', `Chain "${chainName}" failed at step ${i + 1}: ${errMsg.slice(0, 100)}`);

      await reportToCoordinator(
        'main',
        `Chain "${chainName}" failed at step ${i + 1} (${step.agent_id}) after ${MAX_RETRIES + 1} attempts: ${errMsg.slice(0, 200)}`,
        'alert',
      );
      throw lastErr;
    }
  }

  // Save full result to file if it exceeds DB column limit (prevents truncation)
  let resultRef = fullLastResult.slice(0, 2000);
  if (fullLastResult.length > 2000) {
    const resultsDir = path.join(PROJECT_ROOT, 'data', 'chain-results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const resultFile = path.join(resultsDir, `${chainId}-${Date.now()}.md`);
    fs.writeFileSync(resultFile, `# Chain: ${chainName}\n\n${fullLastResult}`, 'utf-8');
    resultRef = `${fullLastResult.slice(0, 1500)}\n\n[... полный результат: ${resultFile}]`;
    logger.info({ chainId, resultFile, fullSize: fullLastResult.length }, 'Full chain result saved to file');
  }

  updateChainRun(chainId, 'completed', resultRef);
  logToHiveMind('main', chatId, 'chain_completed', `Chain "${chainName}" completed`);

  // Report success
  await reportToCoordinator(
    'main',
    `Chain "${chainName}" completed.\n\nResult:\n${prevResult.slice(0, 500)}`,
    'result',
  );

  return prevResult;
}

/**
 * Check all enabled chains against cron schedule and run due ones.
 * Called from scheduler interval in index.ts.
 */
export function checkAndRunChains(chatId: string): void {
  const chains = getScheduledChains(true);
  const now = new Date();

  for (const chain of chains) {
    if (chain.last_status === 'running') continue; // already running

    // Simple cron check: compare HH:MM with chain.cron
    // Supports formats: "HH:MM" (daily) or standard cron
    const timeMatch = chain.cron.match(/^(\d{2}):(\d{2})$/);
    if (timeMatch) {
      const hour = parseInt(timeMatch[1], 10);
      const minute = parseInt(timeMatch[2], 10);
      // Use configured timezone
      const nowHour = (now.getUTCHours() + 4) % 24;
      const nowMinute = now.getUTCMinutes();

      if (nowHour !== hour || nowMinute !== minute) continue;

      // Don't re-run if already ran this minute
      if (chain.last_run_at) {
        const lastRun = new Date(chain.last_run_at + 'Z');
        const diffMs = now.getTime() - lastRun.getTime();
        if (diffMs < 60_000) continue;
      }

      logger.info({ chainId: chain.id, name: chain.name }, 'Running scheduled chain');
      void runChain(chain.id, chain.name, chain.steps, chatId).catch((err) =>
        logger.error({ err, chainId: chain.id }, 'Chain execution failed'),
      );
    }
  }
}
