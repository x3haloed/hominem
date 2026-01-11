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

