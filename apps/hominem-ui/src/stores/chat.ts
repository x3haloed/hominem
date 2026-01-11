import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { agentChat, getDefaultAgentBaseUrl, type AgentMessage } from '@/api/agent'

const LS_AGENT_BASE_URL = 'hominem_agent_base_url'
const LS_SESSION_ID = 'hominem_session_id'

function readLocalStorage(key: string): string {
  try {
    return (localStorage.getItem(key) || '').trim()
  } catch {
    return ''
  }
}

function writeLocalStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, (value || '').trim())
  } catch {
    // ignore
  }
}

export const useChatStore = defineStore('chat', () => {
  const agentBaseUrl = ref(readLocalStorage(LS_AGENT_BASE_URL) || getDefaultAgentBaseUrl())
  const sessionId = ref(readLocalStorage(LS_SESSION_ID))
  const sending = ref(false)
  const error = ref<string>('')
  const messages = ref<AgentMessage[]>([])

  const hasSession = computed(() => !!sessionId.value)

  function setAgentBaseUrl(value: string) {
    agentBaseUrl.value = (value || '').trim()
    writeLocalStorage(LS_AGENT_BASE_URL, agentBaseUrl.value)
  }

  function resetSession() {
    sessionId.value = ''
    writeLocalStorage(LS_SESSION_ID, '')
    messages.value = []
    error.value = ''
  }

  function clearMessages() {
    messages.value = []
    error.value = ''
  }

  async function sendMessage(userText: string) {
    const text = (userText || '').trim()
    if (!text || sending.value) return

    sending.value = true
    error.value = ''

    // Optimistic UI: add the user message immediately.
    messages.value = [...messages.value, { role: 'user', content: text }]

    try {
      const resp = await agentChat({
        agentBaseUrl: agentBaseUrl.value,
        sessionId: sessionId.value || null,
        message: text,
      })
      sessionId.value = (resp.session_id || '').trim()
      writeLocalStorage(LS_SESSION_ID, sessionId.value)
      messages.value = Array.isArray(resp.messages) ? resp.messages : messages.value
    } catch (e: any) {
      const msg = String(e?.message || e || 'Unknown error')
      error.value = msg
      messages.value = [...messages.value, { role: 'assistant', content: `Error: ${msg}` }]
    } finally {
      sending.value = false
    }
  }

  return {
    agentBaseUrl,
    sessionId,
    hasSession,
    sending,
    error,
    messages,
    setAgentBaseUrl,
    resetSession,
    clearMessages,
    sendMessage,
  }
})
