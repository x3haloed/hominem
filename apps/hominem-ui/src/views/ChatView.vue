<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'

import { useChatStore } from '@/stores/chat'

const chat = useChatStore()

const composer = ref<HTMLTextAreaElement | null>(null)
const messagesEl = ref<HTMLDivElement | null>(null)

const draft = ref('')
const statusText = computed(() => (chat.sending ? 'Sending…' : chat.error ? chat.error : ''))
const reasoningOpen = ref<Record<number, boolean>>({})

function roleLabel(role: string): string {
  const r = (role || '').toLowerCase()
  if (r === 'assistant') return 'Assistant'
  if (r === 'user') return 'You'
  if (r === 'tool') return 'Tool'
  if (r === 'system') return 'System'
  return role || 'Message'
}

function isUser(role: string): boolean {
  return (role || '').toLowerCase() === 'user'
}

function formatContent(content: unknown): string {
  if (content == null) return ''
  if (typeof content === 'string') return content
  try {
    return JSON.stringify(content, null, 2)
  } catch {
    return String(content)
  }
}

function hasReasoning(msg: any): boolean {
  const r = msg?.reasoning_content
  if (r == null) return false
  if (typeof r === 'string') return r.trim().length > 0
  return true
}

function toggleReasoning(idx: number) {
  reasoningOpen.value[idx] = !reasoningOpen.value[idx]
}

async function scrollToBottom() {
  await nextTick()
  const el = messagesEl.value
  if (!el) return
  el.scrollTop = el.scrollHeight
}

watch(
  () => chat.messages.length,
  () => scrollToBottom(),
)

async function handleSend() {
  const text = (draft.value || '').trim()
  if (!text) return
  draft.value = ''
  await chat.sendMessage(text)
  await nextTick()
  composer.value?.focus()
}

function onComposerKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend().catch(() => {})
  }
}

function newChat() {
  chat.resetSession()
  draft.value = ''
  reasoningOpen.value = {}
  nextTick(() => composer.value?.focus())
}

onMounted(() => {
  composer.value?.focus()
})
</script>

<template>
  <div class="flex h-screen">
    <!-- Sidebar -->
    <aside class="w-72 bg-gray-900 flex flex-col">
      <div class="p-4 border-b border-gray-700">
        <div class="flex items-center space-x-3">
          <div
            class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center"
          >
            <i class="fa-solid fa-brain text-white text-sm"></i>
          </div>
          <h1 class="text-white font-semibold text-lg">Hominem</h1>
        </div>
        <button
          class="w-full mt-4 bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center justify-center space-x-2 transition-colors"
          @click="newChat"
        >
          <i class="fa-solid fa-plus text-sm"></i><span>New Chat</span>
        </button>
      </div>

      <div class="flex-1 overflow-y-auto p-4 text-gray-300 text-sm space-y-3">
        <div class="text-xs text-gray-400 uppercase tracking-wide">Connection</div>
        <div class="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-2">
          <div class="text-xs text-gray-400">Agent base URL</div>
          <input
            class="w-full px-3 py-2 rounded-md bg-gray-900 border border-gray-700 text-gray-100 placeholder:text-gray-500 text-sm"
            :value="chat.agentBaseUrl"
            placeholder="(empty = same origin)"
            @change="chat.setAgentBaseUrl(($event.target as HTMLInputElement).value)"
          />
          <div class="text-xs text-gray-500">
            Dev: keep this empty and Vite will proxy <code class="bg-gray-900 px-1 rounded">/api</code>.
          </div>
        </div>

        <div class="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-2">
          <div class="text-xs text-gray-400">Session</div>
          <div class="text-sm text-gray-100 truncate">
            {{ chat.sessionId || '(none)' }}
          </div>
        </div>
      </div>
    </aside>

    <!-- Main -->
    <main class="flex-1 flex flex-col bg-white">
      <header class="px-6 py-4 border-b border-gray-200 bg-white">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-xl font-semibold text-gray-900">AI Assistant</h2>
            <p class="text-sm text-gray-500">hominem-ui → hominem-agent → hominem-infer</p>
          </div>
          <div class="flex items-center space-x-2">
            <button
              class="px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-lg transition-colors text-sm"
              @click="chat.clearMessages()"
            >
              Clear
            </button>
          </div>
        </div>
      </header>

      <section ref="messagesEl" class="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        <div v-for="(msg, idx) in chat.messages" :key="idx" class="flex" :class="isUser(msg.role) ? 'justify-end' : 'justify-start'">
          <div class="max-w-3xl">
            <div
              v-if="isUser(msg.role)"
              class="bg-blue-600 text-white rounded-2xl px-4 py-3 shadow-sm whitespace-pre-wrap"
            >
              {{ formatContent(msg.content) }}
            </div>
            <div
              v-else
              class="text-gray-900"
            >
              <div v-if="msg.role === 'assistant'" class="flex items-start space-x-3">
                <div
                  class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0"
                >
                  <i class="fa-solid fa-brain text-white text-sm"></i>
                </div>
                <div class="flex-1">
                  <div
                    v-if="hasReasoning(msg)"
                    class="mb-4 border border-gray-200 rounded-xl overflow-hidden"
                  >
                    <button
                      class="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 flex items-center justify-between transition-colors"
                      type="button"
                      @click="toggleReasoning(idx)"
                    >
                      <div class="flex items-center space-x-2">
                        <i class="fa-solid fa-brain text-purple-600"></i>
                        <span class="font-medium text-gray-700">AI Reasoning</span>
                      </div>
                      <i
                        class="fa-solid fa-chevron-down text-gray-500 transition-transform"
                        :class="reasoningOpen[idx] ? 'rotate-180' : ''"
                      ></i>
                    </button>
                    <div
                      v-show="reasoningOpen[idx]"
                      class="px-4 py-3 bg-purple-50 border-t border-gray-200"
                    >
                      <pre class="text-sm text-gray-700 whitespace-pre-wrap">{{ formatContent(msg.reasoning_content) }}</pre>
                    </div>
                  </div>

                  <div class="bg-gray-100 rounded-2xl px-4 py-3">
                    <pre class="text-gray-800 leading-relaxed whitespace-pre-wrap">{{ formatContent(msg.content) }}</pre>
                  </div>
                </div>
              </div>

              <div v-else class="bg-gray-50 border border-gray-200 text-gray-900 rounded-2xl px-4 py-3 shadow-sm">
                <div class="text-xs text-gray-500 mb-2">{{ roleLabel(msg.role) }}</div>
                <pre class="text-sm leading-relaxed whitespace-pre-wrap">{{ formatContent(msg.content) }}</pre>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer class="px-6 py-4 border-t border-gray-200 bg-white">
        <div class="flex items-end space-x-3">
          <div class="flex-1">
            <textarea
              ref="composer"
              v-model="draft"
              rows="1"
              class="w-full resize-none px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder:text-gray-400"
              placeholder="Message Hominem…"
              :disabled="chat.sending"
              @keydown="onComposerKeydown"
            ></textarea>
            <div class="text-xs text-gray-500 mt-2 min-h-4">
              {{ statusText }}
            </div>
          </div>
          <button
            class="h-11 px-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:hover:bg-blue-600 text-white rounded-xl transition-colors flex items-center space-x-2"
            :disabled="chat.sending"
            @click="handleSend"
          >
            <i class="fa-solid fa-paper-plane text-sm"></i><span class="text-sm font-medium">Send</span>
          </button>
        </div>
      </footer>
    </main>
  </div>
</template>
