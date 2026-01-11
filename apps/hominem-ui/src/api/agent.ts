export type AgentMessage = {
  role: string
  content: unknown
  tool_calls?: unknown
  reasoning_content?: unknown
}

export type ChatResponse = {
  session_id: string
  assistant: string
  messages: AgentMessage[]
}

export type ChatStreamEvent =
  | { type: 'start'; session_id: string }
  | { type: 'assistant'; assistant: AgentMessage }
  | { type: 'done'; session_id: string; assistant: string; messages: AgentMessage[] }
  | { type: 'error'; detail: string }

export function getDefaultAgentBaseUrl(): string {
  // Empty means "same origin" (works with Vite proxy in dev, or when deployed behind a reverse proxy).
  return (import.meta.env.VITE_AGENT_BASE_URL || '').trim()
}

export async function agentChat(params: {
  agentBaseUrl: string
  sessionId: string | null
  message: string
}): Promise<ChatResponse> {
  const base = (params.agentBaseUrl || '').trim()
  const url = `${base}/api/chat`.replace(/\/+api\/chat$/, '/api/chat')

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: params.sessionId,
      message: params.message,
    }),
  })

  let data: any = null
  try {
    data = await resp.json()
  } catch {
    data = null
  }
  if (!resp.ok) {
    const detail = data?.detail || (typeof data === 'string' ? data : null)
    throw new Error(detail || `HTTP ${resp.status}`)
  }
  return data as ChatResponse
}

export async function agentChatStream(params: {
  agentBaseUrl: string
  sessionId: string | null
  message: string
  onEvent: (event: ChatStreamEvent) => void
}): Promise<ChatResponse> {
  const base = (params.agentBaseUrl || '').trim()
  const url = `${base}/api/chat/stream`.replace(/\/+api\/chat\/stream$/, '/api/chat/stream')

  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: params.sessionId,
      message: params.message,
    }),
  })

  if (!resp.ok) {
    let data: any = null
    try {
      data = await resp.json()
    } catch {
      data = null
    }
    const detail = data?.detail || (typeof data === 'string' ? data : null)
    throw new Error(detail || `HTTP ${resp.status}`)
  }

  const body = resp.body
  if (!body) throw new Error('No response body')

  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    while (true) {
      const nl = buffer.indexOf('\n')
      if (nl === -1) break
      const line = buffer.slice(0, nl).trim()
      buffer = buffer.slice(nl + 1)
      if (!line) continue

      const event = JSON.parse(line) as ChatStreamEvent
      params.onEvent(event)
      if (event.type === 'error') {
        throw new Error(event.detail || 'Unknown error')
      }
      if (event.type === 'done') {
        return { session_id: event.session_id, assistant: event.assistant, messages: event.messages }
      }
    }
  }

  throw new Error('Stream ended unexpectedly')
}
