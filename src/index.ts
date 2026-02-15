import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  ASSISTANT_NAME,
  CONTAINER_RUNTIME,
  DATA_DIR,
  GROUPS_DIR,
  HEAP_LIMIT_MB,
  IDLE_TIMEOUT,
  MAIN_GROUP_FOLDER,
  MATRIX_ACCESS_TOKEN,
  MATRIX_HOMESERVER,
  MATRIX_PASSWORD,
  MATRIX_RECONNECT_INTERVAL,
  MATRIX_USERNAME,
  MEMORY_CHECK_INTERVAL,
  POLL_INTERVAL,
  TRIGGER_PATTERN,
} from './config.js';
import { MatrixChannel } from './channels/matrix.js';
import {
  ContainerOutput,
  runContainerAgent,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  getAllChats,
  getAllRegisteredGroups,
  getAllSessions,
  getAllTasks,
  deleteSession,
  getMessagesSince,
  getNewMessages,
  getRouterState,
  initDatabase,
  setRegisteredGroup,
  setRouterState,
  setSession,
  storeChatMetadata,
  storeMessage,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { startIpcWatcher } from './ipc.js';
import { findChannel, formatMessages, stripInternalTags } from './router.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

let lastTimestamp = '';
let sessions: Record<string, string> = {};
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
let messageLoopRunning = false;
const QUEUED_ACK_COOLDOWN_MS = 30_000;
const lastQueuedAckAt: Record<string, number> = {};
const STATUS_REQUEST_PATTERN = /\b(progress|status|update|report)\b/i;
const PROJECT_ENV_PATH = path.join(process.cwd(), '.env');

function firstSet(...values: Array<string | undefined>): string | undefined {
  for (const v of values) {
    const s = v?.trim();
    if (s) return s;
  }
  return undefined;
}

function parseEnvLine(line: string): [string, string] | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith('#')) return null;
  const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
  if (!match) return null;
  const key = match[1];
  let value = match[2];
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    value = value.slice(1, -1);
  }
  return [key, value];
}

function loadProjectEnv(): Record<string, string> {
  const values: Record<string, string> = {};
  if (!fs.existsSync(PROJECT_ENV_PATH)) return values;

  try {
    const envContent = fs.readFileSync(PROJECT_ENV_PATH, 'utf-8');
    for (const line of envContent.split('\n')) {
      const parsed = parseEnvLine(line);
      if (!parsed) continue;
      const [key, value] = parsed;
      values[key] = value;
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to read project .env');
  }

  return values;
}

const PROJECT_ENV = loadProjectEnv();

function getConfiguredEnv(key: string): string | undefined {
  return firstSet(process.env[key], PROJECT_ENV[key]);
}

function resolveConfiguredMainModel(): string | undefined {
  return getConfiguredEnv('NANOCLAW_MAIN_MODEL')?.trim() || undefined;
}

function parseNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function getClaudeModelFromStatsCache(): string | undefined {
  const statsPath = path.join(
    DATA_DIR,
    'sessions',
    MAIN_GROUP_FOLDER,
    '.claude',
    'stats-cache.json',
  );

  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(statsPath, 'utf-8'));
  } catch {
    return undefined;
  }

  if (!parsed || typeof parsed !== 'object') return undefined;

  // Prefer modelUsage since it summarizes overall token usage by model.
  const modelUsage = (parsed as { modelUsage?: unknown }).modelUsage;
  if (modelUsage && typeof modelUsage === 'object') {
    let bestModel: string | undefined;
    let bestScore = -1;
    for (const [model, usage] of Object.entries(modelUsage)) {
      if (!model.trim() || !usage || typeof usage !== 'object') continue;
      const metrics = usage as Record<string, unknown>;
      const score =
        parseNumber(metrics.inputTokens) +
        parseNumber(metrics.outputTokens) +
        parseNumber(metrics.cacheReadInputTokens) +
        parseNumber(metrics.cacheCreationInputTokens);
      if (score > bestScore) {
        bestScore = score;
        bestModel = model.trim();
      }
    }
    if (bestModel) return bestModel;
  }

  // Fallback: inspect most recent daily tokens by model.
  const dailyModelTokens = (parsed as { dailyModelTokens?: unknown }).dailyModelTokens;
  if (Array.isArray(dailyModelTokens)) {
    for (let i = dailyModelTokens.length - 1; i >= 0; i -= 1) {
      const dayEntry = dailyModelTokens[i];
      if (!dayEntry || typeof dayEntry !== 'object') continue;
      const tokensByModel = (dayEntry as { tokensByModel?: unknown }).tokensByModel;
      if (!tokensByModel || typeof tokensByModel !== 'object') continue;

      let bestModel: string | undefined;
      let bestTokens = -1;
      for (const [model, tokens] of Object.entries(tokensByModel)) {
        const tokenCount = parseNumber(tokens);
        if (model.trim() && tokenCount > bestTokens) {
          bestTokens = tokenCount;
          bestModel = model.trim();
        }
      }
      if (bestModel) return bestModel;
    }
  }

  return undefined;
}

function resolveMainProvider(): 'claude' | 'ollama' {
  const anthropicBaseUrl =
    (getConfiguredEnv('ANTHROPIC_BASE_URL') || '').toLowerCase();
  if (anthropicBaseUrl.includes('ollama')) {
    return 'ollama';
  }
  return 'claude';
}

function isGenericClaudeModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  if (!normalized) return true;

  if (
    normalized === 'default' ||
    normalized === 'opus' ||
    normalized === 'sonnet' ||
    normalized === 'haiku' ||
    normalized === 'claude-opus' ||
    normalized === 'claude-sonnet' ||
    normalized === 'claude-haiku'
  ) {
    return true;
  }

  // Treat family aliases like claude-opus, claude-opus-latest as non-specific.
  // Any model string containing digits is considered specific (e.g. claude-opus-4-6).
  if (/^(claude-)?(opus|sonnet|haiku)(-[a-z._-]+)?$/i.test(normalized) && !/\d/.test(normalized)) {
    return true;
  }

  return false;
}

function normalizeMainLlm(model: string | undefined): string | undefined {
  const trimmed = model?.trim();
  if (!trimmed) return undefined;

  if (resolveMainProvider() !== 'claude') {
    return trimmed;
  }

  if (!isGenericClaudeModel(trimmed)) {
    return trimmed;
  }

  // Try to upgrade generic aliases to a concrete dated model if available.
  const fromStats = getClaudeModelFromStatsCache()?.trim();
  if (fromStats && !isGenericClaudeModel(fromStats)) {
    return fromStats;
  }

  return undefined;
}
function resolveMainLlm(): string {
  // Explicit MAIN model override first.
  const directModel = resolveConfiguredMainModel();
  if (directModel) return directModel;

  // If Anthropic endpoint is redirected to Ollama, prefer Ollama model env.
  const anthropicBaseUrl =
    (getConfiguredEnv('ANTHROPIC_BASE_URL') || '').toLowerCase();
  if (anthropicBaseUrl.includes('ollama')) {
    const ollamaModel = normalizeMainLlm(
      firstSet(getConfiguredEnv('OLLAMA_MODEL'), getConfiguredEnv('MODEL')),
    );
    if (ollamaModel) return ollamaModel;
  }

  // For native Claude sessions, wait for runtime init.model so label is exact.
  if (resolveMainProvider() === 'claude') {
    const statsModel = normalizeMainLlm(getClaudeModelFromStatsCache());
    if (statsModel) return statsModel;
    return 'unknown-model';
  }

  // Generic fallback for other redirected providers.
  return (
    normalizeMainLlm(
      firstSet(
        getConfiguredEnv('OLLAMA_MODEL'),
        getConfiguredEnv('OPENAI_MODEL'),
        getConfiguredEnv('MODEL'),
        getConfiguredEnv('GEMINI_MODEL'),
        getConfiguredEnv('CODEX_MODEL'),
      ),
    ) || 'unknown-model'
  );
}
const MAIN_PROVIDER = resolveMainProvider();
let mainLlm = resolveMainLlm();

function updateMainLlm(model?: string): void {
  const normalized = normalizeMainLlm(model);
  if (!normalized || normalized === mainLlm) return;
  mainLlm = normalized;
  setRouterState('main_model', mainLlm);
  logger.info({ mainModel: mainLlm }, 'Updated MAIN model label');
}

function mainSender(): string {
  return `MAIN(${MAIN_PROVIDER},${mainLlm})`;
}

let channels: Channel[] = [];
const queue = new GroupQueue();

function loadState(): void {
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch {
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  const configuredMainModel = resolveConfiguredMainModel();
  const storedMainModel = normalizeMainLlm(getRouterState('main_model'));
  if (configuredMainModel) {
    const pinnedChanged =
      storedMainModel && configuredMainModel !== storedMainModel;
    mainLlm = configuredMainModel;
    setRouterState('main_model', mainLlm);

    // If model pin changed, drop the main session so Claude initializes fresh
    // on the requested model instead of resuming a prior-model session.
    if (pinnedChanged && sessions[MAIN_GROUP_FOLDER]) {
      deleteSession(MAIN_GROUP_FOLDER);
      delete sessions[MAIN_GROUP_FOLDER];
      logger.info(
        {
          fromModel: storedMainModel,
          toModel: configuredMainModel,
        },
        'Pinned MAIN model changed; cleared main session',
      );
    }
  } else if (storedMainModel) {
    mainLlm = storedMainModel;
  }
  logger.info(
    { groupCount: Object.keys(registeredGroups).length, mainModel: mainLlm },
    'State loaded',
  );
}

function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState(
    'last_agent_timestamp',
    JSON.stringify(lastAgentTimestamp),
  );
  setRouterState('main_model', mainLlm);
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  const groupDir = path.join(DATA_DIR, '..', 'groups', group.folder);
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

/**
 * Get available groups list for the agent.
 * Returns groups ordered by most recent activity.
 */
export function getAvailableGroups(): import('./container-runner.js').AvailableGroup[] {
  const chats = getAllChats();
  const registeredJids = new Set(Object.keys(registeredGroups));

  return chats
    .filter((c) => c.jid !== '__group_sync__' && c.jid.startsWith('matrix:'))
    .map((c) => ({
      jid: c.jid,
      name: c.name,
      lastActivity: c.last_message_time,
      isRegistered: registeredJids.has(c.jid),
    }));
}

/** @internal - exported for testing */
export function _setRegisteredGroups(groups: Record<string, RegisteredGroup>): void {
  registeredGroups = groups;
}

/**
 * Append a brief entry to the group's conversation log.
 * Used for cross-channel context between WhatsApp and terminal sessions.
 */
function appendConversationLog(
  groupFolder: string,
  userMessages: NewMessage[],
  agentResponses: string[],
  channelName = 'matrix',
): void {
  if (userMessages.length === 0 && agentResponses.length === 0) return;

  const logDir = path.join(GROUPS_DIR, groupFolder, 'conversations');
  const logPath = path.join(logDir, 'log.md');
  fs.mkdirSync(logDir, { recursive: true });

  const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 16);
  const lines = ['---', `${timestamp} [${channelName}]`];

  for (const msg of userMessages) {
    const content = msg.content.length > 200
      ? msg.content.slice(0, 200) + '...'
      : msg.content;
    lines.push(`${msg.sender_name}: ${content}`);
  }

  for (const response of agentResponses) {
    const content = response.length > 200
      ? response.slice(0, 200) + '...'
      : response;
    lines.push(`${ASSISTANT_NAME}: ${content}`);
  }

  fs.appendFileSync(logPath, lines.join('\n') + '\n');

  // Trim to last 100 entries
  try {
    const full = fs.readFileSync(logPath, 'utf-8');
    const entries = full.split(/(?=^---$)/m).filter((e) => e.trim());
    if (entries.length > 100) {
      fs.writeFileSync(logPath, entries.slice(-100).join(''));
    }
  } catch {
    // Non-critical — log continues to grow until next successful trim
  }
}

/**
 * Process all pending messages for a group.
 * Called by the GroupQueue when it's this group's turn.
 */
async function processGroupMessages(chatJid: string): Promise<boolean> {
  const group = registeredGroups[chatJid];
  if (!group) return true;

  const isMainGroup = group.folder === MAIN_GROUP_FOLDER;

  const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
  const missedMessages = getMessagesSince(chatJid, sinceTimestamp, ASSISTANT_NAME);

  if (missedMessages.length === 0) return true;

  // For non-main groups, check if trigger is required and present
  if (!isMainGroup && group.requiresTrigger !== false) {
    const hasTrigger = missedMessages.some((m) =>
      TRIGGER_PATTERN.test(m.content.trim()),
    );
    if (!hasTrigger) return true;
  }

  const prompt = formatMessages(missedMessages);

  // Advance cursor so the piping path in startMessageLoop won't re-fetch
  // these messages. Save the old cursor so we can roll back on error.
  const previousCursor = lastAgentTimestamp[chatJid] || '';
  lastAgentTimestamp[chatJid] =
    missedMessages[missedMessages.length - 1].timestamp;
  saveState();

  logger.info(
    { group: group.name, messageCount: missedMessages.length },
    'Processing messages',
  );

  // Track idle timer for closing stdin when agent is idle
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      logger.debug({ group: group.name }, 'Idle timeout, closing container stdin');
      queue.closeStdin(chatJid);
    }, IDLE_TIMEOUT);
  };

  const channel = findChannel(channels, chatJid);
  if (channel?.setTyping) await channel.setTyping(chatJid, true);
  let hadError = false;
  let outputSentToUser = false;
  const agentResponses: string[] = [];

  const runResult = await runAgent(group, prompt, chatJid, async (result) => {
    // Streaming output callback — called for each agent result
    if (group.folder === MAIN_GROUP_FOLDER && result.model) {
      updateMainLlm(result.model);
    }
    if (result.result) {
      const raw = typeof result.result === 'string' ? result.result : JSON.stringify(result.result);
      // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
      const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
      logger.info({ group: group.name }, `Agent output: ${raw.slice(0, 200)}`);
      if (text) {
        const ch = findChannel(channels, chatJid);
        if (ch) {
          const prefixed = `${mainSender()}: ${text}`;
          await ch.sendMessage(chatJid, prefixed);
        }
        outputSentToUser = true;
        agentResponses.push(`${mainSender()}: ${text}`);
      }
      // Only reset idle timer on actual results, not session-update markers (result: null)
      resetIdleTimer();
    }

    if (result.status === 'error') {
      hadError = true;
    }
  });

  if (channel?.setTyping) await channel.setTyping(chatJid, false);
  if (idleTimer) clearTimeout(idleTimer);

  if (runResult.status === 'error' || hadError) {
    const rawError =
      runResult.error ||
      (hadError ? 'agent returned an error status' : 'unknown error');
    const compactError = rawError.replace(/\s+/g, ' ').slice(0, 220);

    if (!outputSentToUser && channel) {
      const errorReply =
        `${mainSender()}: I hit an error while processing that request: ${compactError}`;
      try {
        await channel.sendMessage(chatJid, errorReply);
        outputSentToUser = true;
        agentResponses.push(errorReply);
      } catch (sendErr) {
        logger.warn(
          { group: group.name, err: sendErr },
          'Failed to send error reply to channel',
        );
      }
    }

    // If we already sent output to the user, don't roll back the cursor —
    // the user got their response and re-processing would send duplicates.
    if (outputSentToUser) {
      logger.warn({ group: group.name }, 'Agent error after output was sent, skipping cursor rollback to prevent duplicates');
      appendConversationLog(group.folder, missedMessages, agentResponses, channel?.name);
      return true;
    }
    // Roll back cursor so retries can re-process these messages
    lastAgentTimestamp[chatJid] = previousCursor;
    saveState();
    logger.warn({ group: group.name }, 'Agent error, rolled back message cursor for retry');
    return false;
  }

  appendConversationLog(group.folder, missedMessages, agentResponses, channel?.name);
  return true;
}

async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<{ status: 'success' | 'error'; error?: string }> {
  const isMain = group.folder === MAIN_GROUP_FOLDER;
  const sessionKey = group.folder;
  const sessionId = sessions[sessionKey];

  // Update tasks snapshot for container to read (filtered by group)
  const tasks = getAllTasks();
  writeTasksSnapshot(
    group.folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
  );

  // Wrap onOutput to track session ID from streamed results
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId) {
          sessions[sessionKey] = output.newSessionId;
          setSession(sessionKey, output.newSessionId);
        }
        await onOutput(output);
      }
    : undefined;

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
      },
      (proc, containerName) => queue.registerProcess(chatJid, proc, containerName, group.folder),
      wrappedOnOutput,
    );

    if (output.newSessionId) {
      sessions[sessionKey] = output.newSessionId;
      setSession(sessionKey, output.newSessionId);
    }

    if (output.status === 'error') {
      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      return {
        status: 'error',
        error: output.error || 'container agent error',
      };
    }

    return { status: 'success' };
  } catch (err) {
    logger.error({ group: group.name, err }, 'Agent error');
    return {
      status: 'error',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function startMessageLoop(): Promise<void> {
  if (messageLoopRunning) {
    logger.debug('Message loop already running, skipping duplicate start');
    return;
  }
  messageLoopRunning = true;

  logger.info(`NanoClaw running (trigger: @${ASSISTANT_NAME})`);

  while (true) {
    try {
      const jids = Object.keys(registeredGroups);
      const { messages, newTimestamp } = getNewMessages(jids, lastTimestamp, ASSISTANT_NAME);

      if (messages.length > 0) {
        logger.info({ count: messages.length }, 'New messages');

        // Advance the "seen" cursor for all messages immediately
        lastTimestamp = newTimestamp;
        saveState();

        // Deduplicate by group
        const messagesByGroup = new Map<string, NewMessage[]>();
        for (const msg of messages) {
          const existing = messagesByGroup.get(msg.chat_jid);
          if (existing) {
            existing.push(msg);
          } else {
            messagesByGroup.set(msg.chat_jid, [msg]);
          }
        }

        for (const [chatJid, groupMessages] of messagesByGroup) {
          const group = registeredGroups[chatJid];
          if (!group) continue;

          const isMainGroup = group.folder === MAIN_GROUP_FOLDER;
          const needsTrigger = !isMainGroup && group.requiresTrigger !== false;

          // For non-main groups, only act on trigger messages.
          // Non-trigger messages accumulate in DB and get pulled as
          // context when a trigger eventually arrives.
          if (needsTrigger) {
            const hasTrigger = groupMessages.some((m) =>
              TRIGGER_PATTERN.test(m.content.trim()),
            );
            if (!hasTrigger) continue;
          }

          // Pull all messages since lastAgentTimestamp so non-trigger
          // context that accumulated between triggers is included.
          const allPending = getMessagesSince(
            chatJid,
            lastAgentTimestamp[chatJid] || '',
            ASSISTANT_NAME,
          );
          const messagesToSend =
            allPending.length > 0 ? allPending : groupMessages;
          const statusMessages = messagesToSend.filter((m) =>
            STATUS_REQUEST_PATTERN.test(m.content.trim()),
          );
          const nonStatusMessages = messagesToSend.filter(
            (m) => !STATUS_REQUEST_PATTERN.test(m.content.trim()),
          );
          const hasStatusRequest = statusMessages.length > 0;

          if (hasStatusRequest) {
            const now = Date.now();
            if (
              !lastQueuedAckAt[chatJid] ||
              now - lastQueuedAckAt[chatJid] >= QUEUED_ACK_COOLDOWN_MS
            ) {
              const snapshot = queue.getGroupStatus(chatJid);
              const statusLine = snapshot.active
                ? `observed status: active run in progress${snapshot.pendingTasks > 0 ? ` (${snapshot.pendingTasks} queued task${snapshot.pendingTasks === 1 ? '' : 's'})` : ''}.`
                : snapshot.pendingMessages || snapshot.pendingTasks > 0 || snapshot.waitingForSlot
                  ? 'observed status: work is queued and waiting to run.'
                  : 'observed status: no active run right now.';
              const ch = findChannel(channels, chatJid);
              if (ch) {
                try {
                  await ch.sendMessage(
                    chatJid,
                    `${mainSender()}: ${statusLine}`,
                  );
                  lastQueuedAckAt[chatJid] = now;
                } catch (err) {
                  logger.warn(
                    { chatJid, err },
                    'Failed to send observed status acknowledgement',
                  );
                }
              }
            }
          }

          // Status-only prompts are answered by observed state and should not
          // be piped into active runs.
          if (nonStatusMessages.length === 0) {
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            continue;
          }

          const formatted = formatMessages(nonStatusMessages);

          if (queue.sendMessage(chatJid, formatted)) {
            logger.debug(
              { chatJid, count: nonStatusMessages.length },
              'Piped messages to active container',
            );
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            // Show typing indicator while the container processes the piped message
            const ch = findChannel(channels, chatJid);
            if (ch?.setTyping) await ch.setTyping(chatJid, true);
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
            const now = Date.now();
            if (
              !lastQueuedAckAt[chatJid] ||
              now - lastQueuedAckAt[chatJid] >= QUEUED_ACK_COOLDOWN_MS
            ) {
              const ch = findChannel(channels, chatJid);
              if (ch) {
                try {
                  await ch.sendMessage(
                    chatJid,
                    hasStatusRequest
                      ? `${mainSender()}: status request received. I am starting the run now and will send a concrete update shortly.`
                      : `${mainSender()}: working on it now...`,
                  );
                  lastQueuedAckAt[chatJid] = now;
                } catch (err) {
                  logger.warn(
                    { chatJid, err },
                    'Failed to send queued acknowledgement',
                  );
                }
              }
            }
          }
        }
      }
    } catch (err) {
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
    const pending = getMessagesSince(chatJid, sinceTimestamp, ASSISTANT_NAME);
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  if (CONTAINER_RUNTIME === 'podman') {
    try {
      execSync('podman info', { stdio: 'pipe' });
      logger.debug('Podman runtime available');
    } catch (err) {
      logger.error({ err }, 'Podman runtime unavailable');
      console.error(
        '\n╔════════════════════════════════════════════════════════════════╗',
      );
      console.error(
        '║  FATAL: Podman is required but unavailable                     ║',
      );
      console.error(
        '║                                                                ║',
      );
      console.error(
        '║  Ensure podman machine is running and podman is in PATH.       ║',
      );
      console.error(
        '║  Then restart NanoClaw.                                        ║',
      );
      console.error(
        '╚════════════════════════════════════════════════════════════════╝\n',
      );
      throw new Error('Podman is required but not available');
    }

    try {
      const output = execSync('podman ps --format json', {
        stdio: ['pipe', 'pipe', 'pipe'],
        encoding: 'utf-8',
      });
      const containers: Array<{ Names?: string[] | string }> = JSON.parse(
        output || '[]',
      );
      const names = containers.flatMap((c) => {
        if (Array.isArray(c.Names)) return c.Names;
        return c.Names ? [c.Names] : [];
      });
      const orphans = names.filter((n) => n.startsWith('nanoclaw-'));
      for (const name of orphans) {
        try {
          execSync(`podman stop ${name}`, { stdio: 'pipe' });
        } catch {
          // Best-effort cleanup
        }
      }
      if (orphans.length > 0) {
        logger.info({ count: orphans.length, names: orphans }, 'Stopped orphaned podman containers');
      }
    } catch (err) {
      logger.warn({ err }, 'Failed to clean up orphaned podman containers');
    }
    return;
  }

  try {
    execSync('container system status', { stdio: 'pipe' });
    logger.debug('Apple Container system already running');
  } catch {
    logger.info('Starting Apple Container system...');
    try {
      execSync('container system start', { stdio: 'pipe', timeout: 30000 });
      logger.info('Apple Container system started');
    } catch (err) {
      logger.error({ err }, 'Failed to start Apple Container system');
      console.error(
        '\n╔════════════════════════════════════════════════════════════════╗',
      );
      console.error(
        '║  FATAL: Apple Container system failed to start                 ║',
      );
      console.error(
        '║                                                                ║',
      );
      console.error(
        '║  Agents cannot run without Apple Container system.             ║',
      );
      console.error(
        '║  Install from: https://github.com/apple/container              ║',
      );
      console.error(
        '║  Then run: container system start                              ║',
      );
      console.error(
        '║  Then restart NanoClaw                                         ║',
      );
      console.error(
        '╚════════════════════════════════════════════════════════════════╝\n',
      );
      throw new Error('Apple Container system is required but not running');
    }
  }

  // Kill and clean up orphaned NanoClaw containers from previous runs
  try {
    const output = execSync('container ls --format json', {
      stdio: ['pipe', 'pipe', 'pipe'],
      encoding: 'utf-8',
    });
    const containers: { status: string; configuration: { id: string } }[] = JSON.parse(output || '[]');
    const orphans = containers
      .filter((c) => c.status === 'running' && c.configuration.id.startsWith('nanoclaw-'))
      .map((c) => c.configuration.id);
    for (const name of orphans) {
      try {
        execSync(`container stop ${name}`, { stdio: 'pipe' });
      } catch { /* already stopped */ }
    }
    if (orphans.length > 0) {
      logger.info({ count: orphans.length, names: orphans }, 'Stopped orphaned containers');
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to clean up orphaned containers');
  }
}

async function main(): Promise<void> {
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');
  loadState();

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    await queue.shutdown(10000);
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Create Matrix channel (activates only if configured)
  let matrix: MatrixChannel | null = null;
  if (
    MATRIX_HOMESERVER &&
    (MATRIX_ACCESS_TOKEN || (MATRIX_USERNAME && MATRIX_PASSWORD))
  ) {
    matrix = new MatrixChannel({
      onMessage: (_chatJid, msg) => storeMessage(msg),
      onChatMetadata: (chatJid, timestamp, name) => storeChatMetadata(chatJid, timestamp, name),
      registeredGroups: () => registeredGroups,
    });
  }

  // Build channels array (only include connected channels)
  const allChannels: (Channel | null)[] = [matrix];
  const refreshConnectedChannels = () => {
    channels = allChannels.filter((ch): ch is Channel => ch != null && ch.isConnected());
  };
  refreshConnectedChannels();

  // Connect channels
  if (matrix) {
    try {
      await matrix.connect();
    } catch (err) {
      logger.error({ err }, 'Initial Matrix connection failed; continuing in degraded mode');
    }
    refreshConnectedChannels();

    let matrixReconnectInProgress = false;
    setInterval(async () => {
      if (!matrix || matrixReconnectInProgress) return;
      matrixReconnectInProgress = true;
      try {
        const healthy = await matrix.checkHealth();
        if (!healthy) {
          await matrix.connect();
          if (matrix.isConnected()) {
            logger.info('Matrix reconnected');
          }
        }
      } catch (err) {
        logger.warn({ err }, 'Matrix reconnect attempt failed');
      } finally {
        refreshConnectedChannels();
        matrixReconnectInProgress = false;
      }
    }, MATRIX_RECONNECT_INTERVAL);
  }

  // Memory watchdog — gracefully recycle before OOM
  const heapLimitBytes = HEAP_LIMIT_MB * 1024 * 1024;
  setInterval(() => {
    const usage = process.memoryUsage();
    const heapMB = Math.round(usage.heapUsed / 1024 / 1024);
    const rssMB = Math.round(usage.rss / 1024 / 1024);
    logger.info({ heapMB, rssMB, limitMB: HEAP_LIMIT_MB }, 'Memory');
    if (usage.heapUsed > heapLimitBytes) {
      logger.warn({ heapMB, limitMB: HEAP_LIMIT_MB }, 'Heap limit exceeded, recycling');
      shutdown('HEAP_LIMIT');
    }
  }, MEMORY_CHECK_INTERVAL);

  // Start subsystems (independently of connection handler)
  startSchedulerLoop({
    registeredGroups: () => registeredGroups,
    getSessions: () => sessions,
    queue,
    onProcess: (groupJid, proc, containerName, groupFolder) => queue.registerProcess(groupJid, proc, containerName, groupFolder),
    sendMessage: async (jid, rawText) => {
      const ch = findChannel(channels, jid);
      if (!ch) return;
      const text = stripInternalTags(rawText);
      if (text) await ch.sendMessage(jid, `${mainSender()}: ${text}`);
    },
  });
  startIpcWatcher({
    sendMessage: (jid, text) => {
      const ch = findChannel(channels, jid);
      if (ch) return ch.sendMessage(jid, text);
      logger.warn({ jid }, 'No channel found for IPC message');
      return Promise.resolve();
    },
    sendImage: (jid, buffer, filename, mimetype, caption) => {
      const ch = findChannel(channels, jid);
      if (ch?.sendImage) return ch.sendImage(jid, buffer, filename, mimetype, caption);
      logger.warn({ jid }, 'No channel with image support found for IPC image');
      return Promise.resolve();
    },
    sendFile: (jid, buffer, filename, mimetype, caption) => {
      const ch = findChannel(channels, jid);
      if (ch?.sendFile) return ch.sendFile(jid, buffer, filename, mimetype, caption);
      logger.warn({ jid }, 'No channel with file support found for IPC file');
      return Promise.resolve();
    },
    registeredGroups: () => registeredGroups,
    registerGroup,
    syncGroupMetadata: async () => {},
    getAvailableGroups,
    writeGroupsSnapshot: (gf, im, ag, rj) => writeGroupsSnapshot(gf, im, ag, rj),
  });
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();
  startMessageLoop();
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname === new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start NanoClaw');
    process.exit(1);
  });
}
