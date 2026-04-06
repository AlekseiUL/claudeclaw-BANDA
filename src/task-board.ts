/**
 * Task Board — файловая доска проекта (Blackboard Pattern).
 *
 * Markdown-файл в data/board.md с текущими задачами и статусами.
 * Агенты обновляют доску при получении/завершении задач.
 * Босс может посмотреть одним взглядом что происходит.
 *
 * Статусы:
 *   📋 НОВАЯ → ⏳ ВЗЯЛ → 🔄 В РАБОТЕ → ✅ ГОТОВО | ❌ ОШИБКА
 */

import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT } from './config.js';
import { logger } from './logger.js';

const BOARD_PATH = path.join(PROJECT_ROOT, 'data', 'board.md');

export interface BoardTask {
  id: string;
  agent: string;
  task: string;
  status: '📋' | '⏳' | '🔄' | '✅' | '❌';
  updatedAt: string;
}

/**
 * Ensure data/ directory and board file exist.
 */
function ensureBoardFile(): void {
  const dir = path.dirname(BOARD_PATH);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  if (!fs.existsSync(BOARD_PATH)) {
    const header = `# 📋 Task Board\n\n_Последнее обновление: ${now()}_\n\n| Статус | Агент | Задача | Время |\n|--------|-------|--------|-------|\n`;
    fs.writeFileSync(BOARD_PATH, header, 'utf-8');
  }
}

function now(): string {
  return new Date().toLocaleString('en-US', { timeZone: 'Europe/Warsaw', hour12: false });
}

/**
 * Parse the board file into structured tasks.
 */
export function readBoard(): BoardTask[] {
  ensureBoardFile();
  const content = fs.readFileSync(BOARD_PATH, 'utf-8');
  const lines = content.split('\n');
  const tasks: BoardTask[] = [];

  for (const line of lines) {
    // Match table rows: | STATUS | AGENT | TASK | TIME |
    const match = line.match(/^\|\s*(📋|⏳|🔄|✅|❌)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$/);
    if (match) {
      tasks.push({
        id: `${match[2].trim()}-${match[3].trim().slice(0, 20)}`,
        status: match[1] as BoardTask['status'],
        agent: match[2].trim(),
        task: match[3].trim(),
        updatedAt: match[4].trim(),
      });
    }
  }

  return tasks;
}

/**
 * Write the full board back to file.
 * Keeps only last 20 completed tasks to prevent file bloat.
 */
function writeBoard(tasks: BoardTask[]): void {
  ensureBoardFile();

  // Split active and completed
  const active = tasks.filter((t) => !['✅', '❌'].includes(t.status));
  const completed = tasks.filter((t) => ['✅', '❌'].includes(t.status));

  // Keep only last 20 completed
  const recentCompleted = completed.slice(-20);
  const allTasks = [...active, ...recentCompleted];

  let content = `# 📋 Task Board\n\n_Последнее обновление: ${now()}_\n\n`;
  content += `| Статус | Агент | Задача | Время |\n`;
  content += `|--------|-------|--------|-------|\n`;

  for (const t of allTasks) {
    content += `| ${t.status} | ${t.agent} | ${t.task} | ${t.updatedAt} |\n`;
  }

  fs.writeFileSync(BOARD_PATH, content, 'utf-8');
  logger.debug({ taskCount: allTasks.length }, 'Board updated');
}

/**
 * Add a new task to the board.
 */
export function boardAddTask(agent: string, task: string, status: BoardTask['status'] = '📋'): void {
  const tasks = readBoard();
  tasks.push({
    id: `${agent}-${Date.now()}`,
    agent,
    task: task.slice(0, 80),
    status,
    updatedAt: now(),
  });
  writeBoard(tasks);
}

/**
 * Update status of a task on the board.
 * Matches by agent name and task substring.
 */
export function boardUpdateTask(agent: string, taskSubstring: string, newStatus: BoardTask['status']): void {
  const tasks = readBoard();
  const task = tasks.find(
    (t) => t.agent === agent && t.task.includes(taskSubstring) && !['✅', '❌'].includes(t.status),
  );

  if (task) {
    task.status = newStatus;
    task.updatedAt = now();
    writeBoard(tasks);
  } else {
    logger.warn({ agent, taskSubstring }, 'Board task not found for update');
  }
}

/**
 * Mark agent's active task as completed.
 */
export function boardCompleteTask(agent: string, taskSubstring?: string): void {
  const tasks = readBoard();
  const task = tasks.find(
    (t) => t.agent === agent && !['✅', '❌'].includes(t.status) &&
      (!taskSubstring || t.task.includes(taskSubstring)),
  );

  if (task) {
    task.status = '✅';
    task.updatedAt = now();
    writeBoard(tasks);
  }
}

/**
 * Mark agent's active task as failed.
 */
export function boardFailTask(agent: string, taskSubstring?: string): void {
  const tasks = readBoard();
  const task = tasks.find(
    (t) => t.agent === agent && !['✅', '❌'].includes(t.status) &&
      (!taskSubstring || t.task.includes(taskSubstring)),
  );

  if (task) {
    task.status = '❌';
    task.updatedAt = now();
    writeBoard(tasks);
  }
}

/**
 * Get a summary of the board state — for quick reporting.
 */
export function boardSummary(): string {
  const tasks = readBoard();
  const active = tasks.filter((t) => !['✅', '❌'].includes(t.status));
  const done = tasks.filter((t) => t.status === '✅').length;
  const failed = tasks.filter((t) => t.status === '❌').length;

  if (active.length === 0 && done === 0 && failed === 0) {
    return 'Доска пуста.';
  }

  let summary = '';
  if (active.length > 0) {
    summary += `Активных: ${active.length}\n`;
    for (const t of active) {
      summary += `  ${t.status} ${t.agent}: ${t.task}\n`;
    }
  }
  if (done > 0) summary += `Выполнено: ${done}\n`;
  if (failed > 0) summary += `Ошибок: ${failed}\n`;

  return summary.trim();
}
