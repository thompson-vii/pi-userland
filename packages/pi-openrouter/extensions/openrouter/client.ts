import type { ActivityResponse } from '@openrouter/sdk/models/index.js';
import type { GetCreditsResponse } from '@openrouter/sdk/models/operations/index.js';
import { OpenRouter } from '@openrouter/sdk/sdk/sdk.js';
import type { ModelsListResponse } from '@openrouter/sdk/models/index.js';
import { UnauthorizedResponseError } from '@openrouter/sdk/models/errors/index.js';
import { execSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

let client: OpenRouter | null = null;

function getClient(): OpenRouter | null {
  if (client) return client;
  const apiKey = getUsageApiKey();
  if (!apiKey) return null;
  client = new OpenRouter({ apiKey });
  return client;
}

export async function getCredits(): Promise<GetCreditsResponse['data'] | null> {
  const client = getClient();
  if (!client) return null;
  try {
    const response = await client.credits.getCredits();
    return response.data;
  } catch (err) {
    throw mapSdkError(err);
  }
}

export async function getActivity(): Promise<ActivityResponse['data'] | null> {
  const client = getClient();
  if (!client) return null;
  try {
    const response = await client.analytics.getUserActivity();
    return response.data;
  } catch (err) {
    throw mapSdkError(err);
  }
}

/**
 * Fetch the authenticated user's model catalog from OpenRouter.
 * Uses the SDK for consistent error handling and retry behavior.
 */
export async function fetchUserModels(): Promise<ModelsListResponse> {
  const key = getModelSyncApiKey();
  if (!key) {
    throw new AuthError(
      'OpenRouter API key not configured. Set OPENROUTER_API_KEY or OPENROUTER_MANAGEMENT_KEY.',
    );
  }

  try {
    const sdkClient = new OpenRouter({ apiKey: key });
    const response = await sdkClient.models.listForUser({ bearer: key }, {});
    const payload = 'result' in response ? response.result : (response as ModelsListResponse);
    return payload as ModelsListResponse;
  } catch (err: unknown) {
    throw mapSdkError(err);
  }
}

/**
 * Check if the OpenRouter API key is configured.
 */
export function isConfigured(): boolean {
  return !!getApiKey();
}

const AUTH_FILENAME = 'auth.json';
const STORED_KEY_COMMAND_TIMEOUT_MS = 10_000;

let commandKeyCache: { command: string; value: string | undefined } | null = null;

function getAuthFilePath(): string {
  // Honor the same agent-directory override Pi itself supports.
  const agentDir =
    process.env['PI_CODING_AGENT_DIR'] ?? join(homedir(), '.pi', 'agent');
  return join(agentDir, AUTH_FILENAME);
}

function nonEmpty(value: string | undefined): string | undefined {
  return value !== undefined && value.trim() !== '' ? value : undefined;
}

/**
 * Resolve a stored key value, running `!command` entries once per process and
 * caching their trimmed stdout, mirroring Pi's own credential resolution.
 */
function resolveStoredKeyValue(rawKey: string): string | undefined {
  const key = rawKey.trim();
  if (!key) return undefined;
  if (!key.startsWith('!')) return key;

  const command = key.slice(1);
  if (commandKeyCache?.command === command) {
    return commandKeyCache.value;
  }

  let value: string | undefined;
  try {
    const output = execSync(command, {
      encoding: 'utf-8',
      timeout: STORED_KEY_COMMAND_TIMEOUT_MS,
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    value = output !== '' ? output : undefined;
  } catch {
    value = undefined;
  }

  commandKeyCache = { command, value };
  return value;
}

/**
 * Read the OpenRouter API key stored by Pi's `/login` flow in auth.json.
 *
 * Supports plain keys and command-sourced keys (`"!command"`), plus the
 * provider-scoped `env` object. Returns undefined when auth.json, the
 * `openrouter` credential, or its key is missing or unreadable.
 */
export function getStoredOpenRouterKey(): string | undefined {
  try {
    const authPath = getAuthFilePath();
    if (!existsSync(authPath)) return undefined;

    const parsed = JSON.parse(readFileSync(authPath, 'utf-8')) as {
      openrouter?: {
        type?: unknown;
        key?: unknown;
        env?: Record<string, unknown>;
      };
    };

    const credential = parsed?.openrouter;
    if (!credential || credential.type !== 'api_key') return undefined;

    // Provider-scoped env values take priority over the credential's key.
    const envKey = credential.env?.['OPENROUTER_API_KEY'];
    const candidate =
      typeof envKey === 'string' && envKey.trim() !== ''
        ? envKey
        : typeof credential.key === 'string'
          ? credential.key
          : undefined;

    return candidate !== undefined ? resolveStoredKeyValue(candidate) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Resolve an OpenRouter key from the process environment.
 * `preferManagement` selects the usage/account ordering.
 */
function resolveEnvKey(preferManagement: boolean): string | undefined {
  const apiKey = nonEmpty(process.env['OPENROUTER_API_KEY']);
  const mgmtKey = nonEmpty(process.env['OPENROUTER_MANAGEMENT_KEY']);
  return preferManagement ? (mgmtKey ?? apiKey) : (apiKey ?? mgmtKey);
}

/**
 * Get the OpenRouter API key from environment.
 */
export function getApiKey(): string | undefined {
  return process.env['OPENROUTER_API_KEY'];
}

/**
 * Get the API key for usage/account endpoints.
 * Prefers OPENROUTER_MANAGEMENT_KEY for full analytics access.
 * Falls back to the credential stored by Pi in auth.json.
 */
export function getUsageApiKey(): string | undefined {
  return resolveEnvKey(true) ?? getStoredOpenRouterKey();
}

/**
 * Get the API key for model sync endpoint.
 * Prefers OPENROUTER_API_KEY but falls back to OPENROUTER_MANAGEMENT_KEY,
 * then to the credential stored by Pi in auth.json.
 */
export function getModelSyncApiKey(): string | undefined {
  return resolveEnvKey(false) ?? getStoredOpenRouterKey();
}

/**
 * Check if model sync is configured.
 */
export function isConfiguredForModelSync(): boolean {
  return !!getModelSyncApiKey();
}

/**
 * Check if usage/account endpoints are configured.
 */
export function isConfiguredForUsage(): boolean {
  return !!getUsageApiKey();
}

/**
 * Map SDK errors to our error types with proper status codes.
 */
function mapSdkError(err: unknown): Error {
  // Handle UnauthorizedResponseError (401)
  if (err instanceof UnauthorizedResponseError) {
    return new ApiError('Unauthorized: Invalid or expired API key', 401);
  }

  // Handle other SDK errors with statusCode
  if (err instanceof Error && 'statusCode' in err) {
    const statusCode = (err as { statusCode: number }).statusCode;
    if (statusCode === 401) {
      return new AuthError(err.message || 'Unauthorized');
    }
    return new ApiError(err.message || 'API error', statusCode);
  }

  // Map error messages to appropriate status codes
  if (err instanceof Error) {
    const message = err.message.toLowerCase();
    if (message.includes('unauthorized')) {
      return new ApiError('Unauthorized: Invalid or expired API key', 401);
    }
    if (message.includes('rate limit') || message.includes('rate limited')) {
      return new ApiError('Rate limited: Too many requests', 429);
    }
    if (
      message.includes('server error') ||
      message.includes('internal') ||
      message.includes('service unavailable')
    ) {
      return new ApiError(err.message || 'Server error', 500);
    }
    return new ApiError(err.message || 'API error', 500);
  }

  return new Error(String(err));
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthError';
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
