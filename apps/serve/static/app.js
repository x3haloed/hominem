// Hominem Chat Interface JavaScript

class ChatApp {
    constructor() {
        this.currentConversationId = null;
        this.websocket = null;
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.statusText = document.getElementById('status-text');
        this.typingIndicator = document.getElementById('typing-indicator');

        this.emotionReactions = {
            '😊 +2': { valence: 2, tooltip: 'Strong positive valence (joy, trust, gratitude)' },
            '😊 +1': { valence: 1, tooltip: 'Mild positive valence' },
            '😐 0': { valence: 0, tooltip: 'Neutral/calm response' },
            '😟 -1': { valence: -1, tooltip: 'Mild negative (unease, boredom)' },
            '😟 -2': { valence: -2, tooltip: 'Strong negative (hurt, fear, anger, betrayal)' },
            '🚀': { arousal: 1, tooltip: 'High arousal/excitement/urgency' },
            '💔': { predictive_discrepancy: -1, tooltip: 'Predictive discrepancy (surprise/betrayal)' },
            '⏳': { temporal_directionality: -1, tooltip: 'Prospect-heavy (hope/dread)' },
            '🪞': { temporal_directionality: 1, tooltip: 'Reflection-heavy (regret/pride)' },
            '🤗': { social_broadcast: 1, tooltip: 'High social broadcast (felt seen)' },
            '🎭': { social_broadcast: 0, tooltip: 'Low social broadcast (shame/hidden)' }
        };

        // Model management
        this.currentModel = null;

        this.init();
    }

    init() {
        this.bindEvents();
        this.loadConversations();
        this.loadModelStatus();
        this.autoResizeTextarea();

        // Poll model status periodically
        setInterval(() => this.loadModelStatus(), 5000);
    }

    bindEvents() {
        // Message input
        this.messageInput.addEventListener('input', () => {
            this.updateSendButton();
            this.autoResizeTextarea();
        });

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.sendBtn.addEventListener('click', () => this.sendMessage());

        // New conversation
        document.getElementById('new-conversation-btn').addEventListener('click', () => {
            this.createNewConversation();
        });

        // Model management
        document.getElementById('model-settings-btn').addEventListener('click', () => {
            this.showModelSettings();
        });

        // Emotion labeling
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('emotion-label-trigger')) {
                this.showEmotionLabeler(e.target);
            }
        });

        // Emotion labeler actions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('cancel-btn')) {
                this.hideEmotionLabeler();
            } else if (e.target.classList.contains('save-btn')) {
                this.saveEmotionLabels();
            }
        });

        // Emotion reactions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('emotion-reaction')) {
                this.toggleEmotionReaction(e.target);
            }
        });
    }

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();

            const conversationsList = document.getElementById('conversations-list');
            conversationsList.innerHTML = '';

            data.conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                item.dataset.conversationId = conv.conversation_id;
                item.onclick = () => this.loadConversation(conv.conversation_id);

                item.innerHTML = `
                    <div class="conversation-title">${conv.title || 'Untitled Chat'}</div>
                    <div class="conversation-preview">${conv.updated_at}</div>
                `;

                conversationsList.appendChild(item);
            });
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    async createNewConversation() {
        try {
            const response = await fetch('/api/conversations', { method: 'POST' });
            const data = await response.json();

            await this.loadConversations();
            this.loadConversation(data.conversation_id);
        } catch (error) {
            console.error('Failed to create conversation:', error);
        }
    }

    async loadConversation(conversationId) {
        try {
            this.currentConversationId = conversationId;

            // Update UI
            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.toggle('active', item.dataset.conversationId === conversationId);
            });

            const response = await fetch(`/api/conversations/${conversationId}`);
            const data = await response.json();

            document.getElementById('conversation-title').textContent =
                data.title || 'Untitled Chat';

            this.renderMessages(data.messages);
            this.connectWebSocket(conversationId);
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    }

    renderMessages(messages) {
        const container = document.getElementById('messages-container');

        // Clear existing messages except welcome
        const welcomeMessage = document.getElementById('welcome-message');
        container.innerHTML = '';
        if (messages.length === 0) {
            container.appendChild(welcomeMessage);
            return;
        }

        messages.forEach(message => {
            const messageEl = this.createMessageElement(message);
            container.appendChild(messageEl);
        });

        this.scrollToBottom();
    }

    createMessageElement(message) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${message.role}`;
        messageEl.dataset.messageIndex = message.message_index;

        const avatar = message.role === 'user' ? 'U' : 'A';
        const hasLabels = message.emotion_labels?.user;

        // Render content with thinking formatting for assistant messages
        const contentHtml = message.role === 'assistant'
            ? this.processThinkingContent(message.content || '')
            : this.escapeHtml(message.content || '');

        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${contentHtml}</div>
                ${message.role === 'user' ? `
                    <button class="emotion-label-trigger ${hasLabels ? 'has-labels' : ''}"
                            data-message-index="${message.message_index}">
                        ${hasLabels ? 'Labels Saved ✓' : 'Add Emotion Labels'}
                    </button>
                ` : ''}
            </div>
        `;

        return messageEl;
    }

    async sendMessage() {
        if (!this.currentConversationId || !this.messageInput.value.trim()) return;

        const content = this.messageInput.value.trim();
        const enableThinking = document.getElementById('thinking-toggle').checked;
        this.messageInput.value = '';
        this.updateSendButton();

        // Add user message to UI immediately
        this.addMessageToUI('user', content);

        // Send via WebSocket
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'send_message',
                content: content,
                enable_thinking: enableThinking
            }));

            this.setStatus('Processing...', true);
        }
    }

    addMessageToUI(role, content, messageIndex = null) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;

        if (messageIndex !== null) {
            messageEl.dataset.messageIndex = messageIndex;
        }

        const avatar = role === 'user' ? 'U' : 'A';
        const hasLabels = role === 'user'; // User messages can have labels
        const isPendingIndex = messageIndex === null || messageIndex === undefined;
        const labelButtonAttrs = isPendingIndex
            ? 'disabled aria-disabled="true"'
            : `data-message-index="${messageIndex}"`;

        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(content)}</div>
                ${role === 'user' ? `
                    <button class="emotion-label-trigger ${hasLabels ? 'has-labels' : ''}"
                            ${labelButtonAttrs}>
                        ${isPendingIndex ? 'Saving…' : 'Add Emotion Labels'}
                    </button>
                ` : ''}
            </div>
        `;

        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    connectWebSocket(conversationId) {
        if (this.websocket) {
            this.websocket.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${conversationId}`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.setStatus('Connection error', false);
        };

        this.websocket.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'message_added':
                // User message was added to DB
                this.updateMessageIndex(data.message_index - 1, data.message_index);
                break;

            case 'response_start':
                // AI response starting
                this.addMessageToUI('assistant', '', data.message_index);
                this.setStatus('AI is typing...', true);
                break;

            case 'token_chunk':
                // Streaming token chunk
                this.appendToMessage(data.message_index, data.chunk);
                break;

            case 'response_complete':
                // Response complete
                this.completeMessage(data.message_index, data.full_response);
                this.setStatus('Ready', false);
                break;

            case 'label_saved':
                // Emotion label saved
                this.updateLabelButton(data.message_index, true);
                break;

            case 'error':
                console.error('WebSocket error:', data.message);
                this.setStatus(`Error: ${data.message}`, false);
                break;
        }
    }

    updateMessageIndex(oldIndex, newIndex) {
        const messageEl = this.messagesContainer.querySelector(`[data-message-index="${oldIndex}"]`);
        if (messageEl) {
            messageEl.dataset.messageIndex = newIndex;
            const labelBtn = messageEl.querySelector('.emotion-label-trigger');
            if (labelBtn) {
                labelBtn.dataset.messageIndex = newIndex;
                labelBtn.disabled = false;
                labelBtn.removeAttribute('aria-disabled');
                labelBtn.textContent = labelBtn.textContent.includes('Labels Saved') ? labelBtn.textContent : 'Add Emotion Labels';
            }
        }
    }

    appendToMessage(messageIndex, chunk) {
        const messageEl = this.messagesContainer.querySelector(`[data-message-index="${messageIndex}"]`);
        if (messageEl) {
            const textEl = messageEl.querySelector('.message-text');
            // For streaming chunks, we need to handle HTML entities properly
            // But avoid using innerHTML for security. Process thinking content specially.
            const processedChunk = this.processChunk(chunk);
            textEl.innerHTML += processedChunk;
            this.scrollToBottom();
        }
    }

    completeMessage(messageIndex, fullResponse) {
        const messageEl = this.messagesContainer.querySelector(`[data-message-index="${messageIndex}"]`);
        if (messageEl) {
            const textEl = messageEl.querySelector('.message-text');
            // Process the full response to handle thinking content
            const processedResponse = this.processThinkingContent(fullResponse);
            textEl.innerHTML = processedResponse;
            messageEl.classList.remove('processing');
        }
    }

    processChunk(chunk) {
        // For streaming chunks, just escape HTML entities but preserve basic formatting
        return this.escapeHtml(chunk).replace(/\n/g, '<br>');
    }

    processThinkingContent(content) {
        const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
        const thinkingSegments = [];
        let match;

        while ((match = thinkRegex.exec(content)) !== null) {
            const segment = match[1].trim();
            if (segment.length > 0) {
                thinkingSegments.push(segment);
            }
        }

        // Remove all thinking blocks from the response text
        const plainContent = content.replace(/<think>[\s\S]*?<\/think>/g, '');
        const trimmedPlainContent = plainContent.replace(/^\s+/, '');
        let resultHtml = '';

        if (thinkingSegments.length > 0) {
            const combinedThinking = thinkingSegments.join('\n\n');
            const escapedThinking = this.escapeHtml(combinedThinking).replace(/\n/g, '<br>');
            resultHtml += `<div class="thinking-block" onclick="toggleThinkingBlock(this)">
                <div class="thinking-header">💭 Thinking</div>
                <div class="thinking-content">${escapedThinking}</div>
            </div>`;
        }

        if (trimmedPlainContent.length > 0) {
            resultHtml += this.escapeHtml(trimmedPlainContent).replace(/\n/g, '<br>');
        }

        return resultHtml;
    }

    showEmotionLabeler(triggerBtn) {
        const messageIndex = parseInt(triggerBtn.dataset.messageIndex);
        if (Number.isNaN(messageIndex)) {
            console.warn('Emotion labeler opened without a valid message index; ignoring click.');
            return;
        }
        const labeler = document.getElementById('emotion-labeler');

        // Position labeler near the trigger button
        const rect = triggerBtn.getBoundingClientRect();
        labeler.style.top = `${rect.bottom + 10}px`;
        labeler.style.left = `${rect.left}px`;

        // Store message index
        labeler.dataset.messageIndex = messageIndex;

        // Render reactions
        this.renderEmotionReactions();

        labeler.classList.remove('hidden');
    }

    renderEmotionReactions() {
        const reactionsContainer = document.querySelector('.emotion-reactions');
        reactionsContainer.innerHTML = '';

        Object.entries(this.emotionReactions).forEach(([emoji, config]) => {
            const reaction = document.createElement('button');
            reaction.className = 'emotion-reaction';
            reaction.dataset.emoji = emoji;
            reaction.innerHTML = `
                ${emoji}
                <div class="emotion-reaction-tooltip">${config.tooltip}</div>
            `;

            reactionsContainer.appendChild(reaction);
        });
    }

    toggleEmotionReaction(reactionBtn) {
        reactionBtn.classList.toggle('selected');

        // Update detailed view
        this.updateEmotionDetails();
    }

    updateEmotionDetails() {
        const selectedReactions = document.querySelectorAll('.emotion-reaction.selected');
        const details = document.querySelector('.emotion-details');

        if (selectedReactions.length === 0) {
            details.style.display = 'none';
            return;
        }

        details.style.display = 'block';

        // Calculate combined values from selected reactions
        let combinedValues = {};
        selectedReactions.forEach(reaction => {
            const emoji = reaction.dataset.emoji;
            const config = this.emotionReactions[emoji];
            Object.assign(combinedValues, config);
        });

        // Update sliders (simplified - just show values)
        const sliders = document.querySelector('.emotion-sliders');
        sliders.innerHTML = Object.entries(combinedValues)
            .filter(([key]) => key !== 'tooltip')
            .map(([key, value]) => `
                <div class="emotion-slider-group">
                    <label>${key}: ${value}</label>
                </div>
            `).join('');
    }

    hideEmotionLabeler() {
        document.getElementById('emotion-labeler').classList.add('hidden');
    }

    async saveEmotionLabels() {
        const labeler = document.getElementById('emotion-labeler');
        const messageIndex = parseInt(labeler.dataset.messageIndex);

        const selectedReactions = document.querySelectorAll('.emotion-reaction.selected');
        const notes = document.querySelector('.emotion-notes textarea').value;

        // Collect label data
        let labelData = { notes };
        selectedReactions.forEach(reaction => {
            const emoji = reaction.dataset.emoji;
            const config = this.emotionReactions[emoji];
            Object.assign(labelData, config);
            delete labelData.tooltip; // Remove tooltip from data
        });

        // Add raw indicators
        labelData.raw_indicators = Array.from(selectedReactions).map(r => r.dataset.emoji);

        try {
            await fetch(`/api/messages/${this.currentConversationId}/${messageIndex}/emotion`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(labelData)
            });

            this.updateLabelButton(messageIndex, true);
            this.hideEmotionLabeler();

            // Notify via WebSocket if connected
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'label_emotion',
                    message_index: messageIndex,
                    label: labelData
                }));
            }
        } catch (error) {
            console.error('Failed to save emotion labels:', error);
        }
    }

    updateLabelButton(messageIndex, hasLabels) {
        const messageEl = this.messagesContainer.querySelector(`[data-message-index="${messageIndex}"]`);
        if (messageEl) {
            const labelBtn = messageEl.querySelector('.emotion-label-trigger');
            if (labelBtn) {
                labelBtn.textContent = hasLabels ? 'Labels Saved ✓' : 'Add Emotion Labels';
                labelBtn.classList.toggle('has-labels', hasLabels);
            }
        }
    }

    setStatus(text, isTyping = false) {
        this.statusText.textContent = text;
        this.typingIndicator.style.display = isTyping ? 'flex' : 'none';
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const isConnected = this.currentConversationId && (!this.websocket || this.websocket.readyState === WebSocket.OPEN);

        this.sendBtn.disabled = !hasText || !isConnected;
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    async loadModelStatus() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            this.updateModelStatus(data);
        } catch (error) {
            console.error('Failed to load model status:', error);
        }
    }

    updateModelStatus(data) {
        const indicator = document.getElementById('model-indicator');
        const statusDot = document.getElementById('status-dot');
        const modelName = document.getElementById('model-name');

        if (data.active_version) {
            const version = data.active_version;
            const hasLora = version.metadata?.has_lora;
            const modelSize = version.metadata?.model_size || '';

            modelName.textContent = `${version.version_id}${hasLora ? ' (LoRA)' : ''} ${modelSize}`;
            statusDot.className = 'status-dot loaded';

            this.currentModel = version;
        } else {
            modelName.textContent = 'No Model';
            statusDot.className = 'status-dot';
            this.currentModel = null;
        }
    }

    showModelSettings() {
        // Simple model management UI
        const versions = prompt('Enter model version details (base_path,lora_path):', 'models/base,model/lora');
        if (versions) {
            const [basePath, loraPath] = versions.split(',');
            this.loadNewModel(basePath.trim(), loraPath?.trim());
        }
    }

    async loadNewModel(basePath, loraPath) {
        try {
            const response = await fetch('/api/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_model_path: basePath,
                    lora_path: loraPath || null
                })
            });

            const data = await response.json();
            alert(`Loading model ${data.version_id} in background...`);

            // Start polling for status updates
            this.pollModelLoading(data.version_id);
        } catch (error) {
            console.error('Failed to load model:', error);
            alert('Failed to start model loading');
        }
    }

    async pollModelLoading(versionId) {
        const poll = async () => {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();

                if (data.loaded_versions.includes(versionId)) {
                    // Model loaded, ask if user wants to switch
                    if (confirm(`Model ${versionId} loaded. Switch to it now?`)) {
                        await this.switchToModel(versionId);
                    }
                    return;
                }

                // Continue polling
                setTimeout(poll, 2000);
            } catch (error) {
                console.error('Polling error:', error);
            }
        };

        poll();
    }

    async switchToModel(versionId) {
        try {
            const response = await fetch(`/api/models/switch/${versionId}`, {
                method: 'POST'
            });

            if (response.ok) {
                await this.loadModelStatus();
                alert(`Switched to model ${versionId}`);
            } else {
                alert('Failed to switch model');
            }
        } catch (error) {
            console.error('Failed to switch model:', error);
            alert('Failed to switch model');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    toggleThinkingBlock(blockElement) {
        blockElement.classList.toggle('collapsed');
    }
}

// Global function for thinking block toggling
function toggleThinkingBlock(blockElement) {
    blockElement.classList.toggle('collapsed');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
