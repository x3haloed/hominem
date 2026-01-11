import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { agentChatStream, getDefaultAgentBaseUrl, type AgentMessage } from '@/api/agent'

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
    // Placeholder assistant message we will update as the stream arrives.
    messages.value = [...messages.value, { role: 'assistant', content: '', reasoning_content: '' }]

    try {
      await agentChatStream({
        agentBaseUrl: agentBaseUrl.value,
        sessionId: sessionId.value || null,
        message: text,
        onEvent: (evt) => {
          if (evt.type === 'start') {
            sessionId.value = (evt.session_id || '').trim()
            writeLocalStorage(LS_SESSION_ID, sessionId.value)
            return
          }
          if (evt.type === 'assistant') {
            // Replace the last assistant message with the latest snapshot.
            const next = [...messages.value]
            for (let i = next.length - 1; i >= 0; i--) {
              if ((next[i]?.role || '').toLowerCase() === 'assistant') {
                next[i] = { ...next[i], ...evt.assistant }
                messages.value = next
                return
              }
            }
            next.push(evt.assistant)
            messages.value = next
            return
          }
          if (evt.type === 'done') {
            sessionId.value = (evt.session_id || '').trim()
            writeLocalStorage(LS_SESSION_ID, sessionId.value)
            messages.value = Array.isArray(evt.messages) ? evt.messages : messages.value
          }
        },
      })
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
