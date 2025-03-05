document.addEventListener('DOMContentLoaded', function () {
    const MAX_CHAT_HISTORY = 5;
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.querySelector('button');
    const paginationContainer = document.getElementById('pagination-container');
    const pdfSection = document.getElementById('pdf-section');
    const pdfViewer = document.getElementById('pdf-viewer');
    const chatSection = document.getElementById('chat-section');
    const closePdfButton = document.getElementById('close-pdf');

    let chatHistory = [];
    let debugLog = [];

    // Authentication related functions
    function disableChatInterface() {
        userInput.disabled = true;
        sendButton.disabled = true;
        chatContainer.style.opacity = '0.5';
        chatContainer.style.pointerEvents = 'none';
    }

    function enableChatInterface() {
        userInput.disabled = false;
        sendButton.disabled = false;
        chatContainer.style.opacity = '1';
        chatContainer.style.pointerEvents = 'auto';
    }

    function checkAuth() {
        const isAuthenticated = sessionStorage.getItem('chatbot_authenticated') === 'true';
        if (isAuthenticated) {
            enableChatInterface();
        } else {
            disableChatInterface();
        }
    }

    // Initial auth check
    checkAuth();

    // Listen for authentication events
    window.addEventListener('authStatusChanged', checkAuth);

    function logDebugInfo(type, info) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            type,
            info
        };
        debugLog.push(logEntry);
        console.log(`[${logEntry.timestamp}] ${type}:`, info);
    }

    function getDebugHeaders(headers) {
        return Object.fromEntries(
            Array.from(headers.entries())
                .filter(([key]) => key.toLowerCase().startsWith('x-debug-'))
        );
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'mb-4 text-left';
        typingDiv.id = 'typing-indicator';
        
        const typingBubble = document.createElement('div');
        typingBubble.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingBubble.appendChild(dot);
        }
        
        typingDiv.appendChild(typingBubble);
        chatContainer.appendChild(typingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    function displayPagination(pageInfo) {
        if (!pageInfo || !pageInfo.total_pages) {
            paginationContainer.classList.add('hidden');
            return;
        }

        logDebugInfo('pagination', pageInfo);
        paginationContainer.classList.remove('hidden');
        paginationContainer.innerHTML = '';

        // Add "Previous" link
        if (pageInfo.current_page > 1) {
            const prevLink = document.createElement('span');
            prevLink.className = 'cursor-pointer hover:underline text-blue-500 mr-4';
            prevLink.textContent = 'Previous';
            prevLink.addEventListener('click', () => {
                handlePaginationClick(`Show me page ${pageInfo.current_page - 1}`);
            });
            paginationContainer.appendChild(prevLink);
        }

        // Add page numbers
        for (let i = 1; i <= pageInfo.total_pages; i++) {
            const pageLink = document.createElement('span');
            pageLink.className = `cursor-pointer hover:underline mx-1 ${i === pageInfo.current_page
                ? 'text-blue-600 font-bold'
                : 'text-blue-500'
                }`;
            pageLink.textContent = i;

            pageLink.addEventListener('click', () => {
                if (i !== pageInfo.current_page) {
                    handlePaginationClick(`Show me page ${i}`);
                }
            });

            paginationContainer.appendChild(pageLink);
        }

        // Add "Next" link
        if (pageInfo.current_page < pageInfo.total_pages) {
            const nextLink = document.createElement('span');
            nextLink.className = 'cursor-pointer hover:underline text-blue-500 ml-4';
            nextLink.textContent = 'Next';
            nextLink.addEventListener('click', () => {
                handlePaginationClick(`Show me page ${pageInfo.current_page + 1}`);
            });
            paginationContainer.appendChild(nextLink);
        }
    }

    function openPdfViewer(url) {
        logDebugInfo('pdf_viewer_open', { url });
        const decodedUrl = decodeURIComponent(url);
        const documentName = decodedUrl.split('/').pop().split('?')[0];

        const pdfTitle = document.getElementById('pdf-title');
        pdfTitle.textContent = documentName;

        // Add transition class before making changes
        chatSection.classList.add('pdf-transition');
        pdfSection.classList.add('pdf-transition');

        pdfViewer.src = url;
        pdfSection.classList.remove('hidden');
        
        // Use requestAnimationFrame to ensure smooth transition
        requestAnimationFrame(() => {
            chatSection.style.width = '50%';
            pdfSection.style.width = '50%';
        });
    }

    function closePdfViewer() {
        logDebugInfo('pdf_viewer_close');
        
        // Add transition class if not already present
        chatSection.classList.add('pdf-transition');
        pdfSection.classList.add('pdf-transition');
    
        // Animate width changes
        chatSection.style.width = '100%';
        pdfSection.style.width = '0';
    
        // Wait for transition to complete before hiding
        setTimeout(() => {
            pdfSection.classList.add('hidden');
            pdfViewer.src = '';
            
            // Clean up transition classes
            chatSection.classList.remove('pdf-transition');
            pdfSection.classList.remove('pdf-transition');
        }, 300); // Match the transition duration
    }

    async function handlePaginationClick(query) {
        logDebugInfo('pagination_click', { query });
        await processMessage(query, false);
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        logDebugInfo('send_message', { message });
        userInput.value = '';
        await processMessage(message, true);
    }

    async function processMessage(message, showQuery) {
        const requestStartTime = Date.now();
        logDebugInfo('request_start', { message, showQuery });

        appendMessage('user', message);
    
        // First add message to chat history
        chatHistory.push({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ]
        });
        chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
    
        showTypingIndicator();
    
        try {
            const requestBody = {
                query: message,
                chats: chatHistory
            };
            logDebugInfo('request_body', requestBody);

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const debugHeaders = getDebugHeaders(response.headers);
            logDebugInfo('response_headers', {
                ...debugHeaders,
                status: response.status,
                statusText: response.statusText
            });
            
            removeTypingIndicator();
            
            const responseText = await response.text();
            logDebugInfo('response_text', {
                length: responseText.length,
                preview: responseText.slice(0, 200)
            });

            // Check if response is empty or contains error indicators
            if (!responseText) {
                logDebugInfo('error_empty_response');
                throw new Error('error_empty_response');
            }

            if (responseText.includes('Backend call failure')) {
                logDebugInfo('backend_call_failure', { responseText });
                throw new Error('The server is currently unavailable. Please try again in a few moments.');
            }

            let data;
            try {
                data = JSON.parse(responseText);
                logDebugInfo('parsed_response', {
                    hasError: !!data.error,
                    debugInfo: data.debug_info
                });
            } catch (parseError) {
                logDebugInfo('parse_error', {
                    error: parseError.message,
                    responsePreview: responseText.slice(0, 200)
                });
                throw new Error('Failed to parse server response');
            }
    
            if (data.error) {
                let errorMessage = data.error;
                if (data.details) {
                    errorMessage += '\n' + data.details;
                }
                
                logDebugInfo('error_response', {
                    error: data.error,
                    details: data.details,
                    debugInfo: data.debug_info
                });
                
                switch(response.status) {
                    case 503:
                        errorMessage = 'Service is temporarily unavailable. Please try again in a few moments.';
                        break;
                    case 504:
                        errorMessage = 'Request timed out. Please try again.';
                        break;
                    case 502:
                        errorMessage = 'Server is temporarily unavailable. Please try again later.';
                        break;
                    case 400:
                        errorMessage = 'Invalid request. Please check your input.';
                        break;
                }
                
                throw new Error(errorMessage);
            }
    
            const result = data.data?.final_result?.output;
            logDebugInfo('process_result', {
                hasResult: !!result,
                status: result?.status,
                hasSummary: !!result?.summary,
                hasReferences: !!result?.references,
                hasPageInfo: !!result?.page_info
            });

            if (result?.status === 'success' && result?.summary) {
                appendMessage('assistant', result.summary, result.references, true);
            
                if (result.page_info) {
                    displayPagination(result.page_info);
                } else {
                    paginationContainer.classList.add('hidden');
                }
            } else if (result?.status === 'error') {
                let errorMessage;
    
                if (result.summary && result.summary.includes('No documents found in document_list')) {
                    errorMessage = 'No documents were found matching your description. Please try:\n\n' +
                                 '1. Using different keywords\n' +
                                 '2. Checking your spelling\n' +
                                 '3. Making your search broader';
                }
                else if (result.summary && result.summary.includes('maximum context length')) {
                    errorMessage = 
                        "Your query requires too much context to process. Please try to:\n\n" +
                        "1. Be more specific in your question\n" +
                        "2. Focus on a particular aspect or topic\n" +
                        "3. Limit the time period or scope of your query";
                } else {
                    errorMessage = result.summary;
                }
                
                appendMessage('system', errorMessage);
                paginationContainer.classList.add('hidden');
            } else {
                throw new Error('Received unexpected response format from server');
            }
    
        } catch (error) {
            removeTypingIndicator();
            logDebugInfo('error', {
                message: error.message,
                stack: error.stack,
                timeTaken: Date.now() - requestStartTime
            });
            
            const errorMessage = error.message || 'An unexpected error occurred. Please try again.';
            appendMessage('system', errorMessage);
            
            paginationContainer.classList.add('hidden');
        }
    }

    function handleTableRows(table) {
        const MAX_VISIBLE_ROWS = 20;
        const rows = table.querySelectorAll('tbody tr');
        
        if (rows.length <= MAX_VISIBLE_ROWS) return; // No need to add show more button
        
        // Hide extra rows
        rows.forEach((row, index) => {
            if (index >= MAX_VISIBLE_ROWS) {
                row.classList.add('table-row-hidden');
            }
        });
        
        // Add fade effect to last visible row
        rows[MAX_VISIBLE_ROWS - 1].classList.add('table-fade-row');
        
        // Create and add the show more/less button
        const button = document.createElement('button');
        button.className = 'show-more-button';
        button.textContent = 'Show All';
        
        let isExpanded = false;
        
        button.addEventListener('click', () => {
            isExpanded = !isExpanded;
            rows.forEach((row, index) => {
                if (index >= MAX_VISIBLE_ROWS) {
                    row.classList.toggle('table-row-hidden');
                }
            });
            
            // Toggle fade effect on the last visible row
            rows[MAX_VISIBLE_ROWS - 1].classList.toggle('table-fade-row');
            
            button.textContent = isExpanded ? 'Show Less' : 'Show All';
        });
        
        // Insert button after the table
        table.parentNode.insertBefore(button, table.nextSibling);
    }

    function appendMessage(role, content, sources = [], isHTML = false) {
        logDebugInfo('append_message', { role, contentLength: content.length, sourcesCount: sources.length });
        
        // Function to estimate tokens in a string
        function estimateTokenCount(text) {
            // Rough estimation: Split by whitespace and punctuation
            const tokens = text.split(/[\s,.!?;:()\[\]{}'"/<>|\\_+=~`@#$%^&*-]+/);
            // Filter out empty strings and count remaining tokens
            return tokens.filter(token => token.length > 0).length;
        }
    
        // Function to trim content to token limit
        function trimToTokenLimit(text, tokenLimit) {
            const lines = text.split('\n');
            let totalTokens = 0;
            let resultLines = [];
            let headerRows = [];
            let isInTable = false;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const lineTokens = estimateTokenCount(line);
                
                // Handle table headers
                if (line.includes('|')) {
                    if (!isInTable) {
                        isInTable = true;
                        headerRows = [line];  // Store first header row
                        continue;
                    } else if (line.includes('|-')) {
                        headerRows.push(line);  // Store separator row
                        continue;
                    }
                } else if (line.trim() === '') {
                    isInTable = false;
                }
                
                // If we're about to exceed the token limit
                if (totalTokens + lineTokens > tokenLimit) {
                    if (isInTable) {
                        // If we're in a table, add the headers before stopping
                        resultLines.push(...headerRows);
                    }
                    resultLines.push('\n... (Content trimmed to fit token limit)');
                    break;
                }
                
                // Add headers if this is the first row after headers
                if (isInTable && headerRows.length > 0) {
                    resultLines.push(...headerRows);
                    headerRows = [];
                }
                
                resultLines.push(line);
                totalTokens += lineTokens;
            }
            
            return resultLines.join('\n');
        }
    
        // Create a trimmed version for chat history
        let chatHistoryContent = content;
        const TOKEN_LIMIT = 20000;
        
        if (role === 'assistant') {
            const estimatedTokens = estimateTokenCount(content);
            if (estimatedTokens > TOKEN_LIMIT) {
                chatHistoryContent = trimToTokenLimit(content, TOKEN_LIMIT);
                logDebugInfo('chat_history_trimmed', {
                    originalTokens: estimatedTokens,
                    newTokens: estimateTokenCount(chatHistoryContent)
                });
            }
        }
        
        // Process display content
        let displayContent = content;
        let isLongResponse = false;
        
        if (role === 'assistant' && estimateTokenCount(content) > TOKEN_LIMIT) {
            isLongResponse = true;
            displayContent += '\n\n---\n*Due to length limitations, I may not have access to all previously mentioned documents and information in our conversation history*';
        }
        
        chatHistory.push({
            "role": role === "user" ? "user" : "assistant",
            "content": [
                {
                    "type": "text",
                    "text": chatHistoryContent
                }
            ]
        });
        chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
    
        const messageDiv = document.createElement('div');
        messageDiv.className = `mb-4 ${role === 'user' ? 'text-right' : 'text-left'}`;
    
        const bubble = document.createElement('div');
        bubble.className = `inline-block space-y-2 p-3 rounded-lg max-w-3/4 ${
            role === 'user'
                ? 'bg-blue-500 text-white'
                : role === 'system'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-300 text-gray-800'
        }`;
    
        // Convert numbered lists in error messages to proper markdown
        if (role === 'system' && displayContent.includes('\n1.')) {
            displayContent = displayContent.split('\n').map(line => {
                if (/^\d+\./.test(line)) {
                    return line.trim();
                }
                return line;
            }).join('\n');
        }
    
        // Add sources if available
        const messageId = crypto.randomUUID();
        let sources_html = document.createElement('div');
        if (sources.length > 0) {
            sources_html.innerHTML = `
                <div class="mt-4 border-t border-gray-200 pt-4">
                    <div data-accordion="open">
                        <h2>
                            <button type="button"
                                class="flex items-center justify-between w-full py-2 px-3 text-sm font-medium text-gray-500 hover:bg-gray-100 rounded-lg gap-3"
                                aria-expanded="false"
                                data-accordion-target="#refs-${messageId}">
                                <span>References (${sources.length})</span>
                                <svg class="w-3 h-3 shrink-0 transition-transform" aria-hidden="true"
                                    xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
                                    <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                        d="M1 1L5 5L9 1"/>
                                </svg>
                            </button>
                        </h2>
                        <div id="refs-${messageId}"
                            class="hidden overflow-hidden transition-[height]"
                            aria-expanded="false">
                            <div class="py-2 px-3 text-sm">
                                <ul class="space-y-2 list-none">
                                    ${sources.map(source => {
                                        const match = source.match(/\[(.*?)\]\((.*?)\)/);
                                        if (match) {
                                            const [_, text, url] = match;
                                            const restOfText = source.split('),')[1] || '';
                                            return `
                                                <li>
                                                    <a href="${url}" class="text-blue-600 hover:underline cursor-pointer" onClick="event.preventDefault(); openPdfViewer('${url}')">
                                                        ${text}
                                                    </a>${restOfText}
                                                </li>
                                            `;
                                        }
                                        return `<li>${source}</li>`;
                                    }).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    
        bubble.innerHTML = marked.parse(displayContent);
        bubble.appendChild(sources_html);
    
        if (isHTML) {
            const links = bubble.getElementsByTagName('a');
            Array.from(links).forEach(link => {
                if (link.href.includes('/api/documents/')) {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        openPdfViewer(link.href);
                    });
                    link.className = 'text-blue-600 hover:underline cursor-pointer';
                }
            });
        }
    
        // Style adjustments
        Array.from(bubble.getElementsByTagName('h5')).forEach(el => {
            el.className += 'text-lg';
        });
        Array.from(bubble.getElementsByTagName('ul')).forEach(el => {
            if (!el.closest('[data-accordion]')) {
                el.className += 'list-inside list-disc';
                el.style.marginTop = '0';
            }
        });
        Array.from(bubble.getElementsByTagName('ol')).forEach(el => {
            el.className = 'list-decimal space-y-2 pl-5';
        });
        Array.from(bubble.getElementsByTagName('li')).forEach(el => {
            el.className = 'pl-2';
            el.style.display = 'list-item';
        });
    
        // Handle tables with show more/less functionality
        Array.from(bubble.getElementsByTagName('table')).forEach(table => {
            table.className = 'table-auto border-collapse bg-white bg-opacity-50 rounded-lg';
            
            // Add table wrapper
            const wrapper = document.createElement('div');
            wrapper.className = 'table-wrapper';
            table.parentNode.insertBefore(wrapper, table);
            wrapper.appendChild(table);
    
            // Handle table rows visibility
            const rows = table.querySelectorAll('tbody tr');
            const MAX_VISIBLE_ROWS = 20;
    
            if (rows.length > MAX_VISIBLE_ROWS) {
                // Hide extra rows
                rows.forEach((row, index) => {
                    if (index >= MAX_VISIBLE_ROWS) {
                        row.classList.add('table-row-hidden');
                    }
                });
    
                // Add fade effect to last visible row
                rows[MAX_VISIBLE_ROWS - 1].classList.add('table-fade-row');
    
                // Create and add the show more/less button
                const button = document.createElement('button');
                button.className = 'show-more-button';
                button.textContent = 'Show All';
    
                let isExpanded = false;
    
                button.addEventListener('click', () => {
                    isExpanded = !isExpanded;
                    rows.forEach((row, index) => {
                        if (index >= MAX_VISIBLE_ROWS) {
                            row.classList.toggle('table-row-hidden');
                        }
                    });
    
                    // Toggle fade effect on the last visible row
                    rows[MAX_VISIBLE_ROWS - 1].classList.toggle('table-fade-row');
                    button.textContent = isExpanded ? 'Show Less' : 'Show All';
                });
    
                // Insert button after the table
                table.parentNode.insertBefore(button, table.nextSibling);
            }
    
            // Apply existing table styles
            Array.from(table.getElementsByTagName('th')).forEach(el => {
                el.className = 'px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider bg-gray-100';
                el.style = '';
            });
            Array.from(table.getElementsByTagName('td')).forEach(el => {
                el.className = 'px-4 py-3 text-sm text-gray-900';
                el.style = '';
            });
        });
    
        messageDiv.appendChild(bubble);
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    
        if (sources.length > 0) {
            initFlowbite();
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    closePdfButton.addEventListener('click', closePdfViewer);

    // Add debug keyboard shortcut (Ctrl+Shift+D)
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.shiftKey && e.key === 'D') {
            console.log('=== Debug Log ===');
            console.table(debugLog);
            
            // Create debug data blob for download
            const debugData = JSON.stringify({
                log: debugLog,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                windowSize: {
                    width: window.innerWidth,
                    height: window.innerHeight
                },
                chatHistory: chatHistory
            }, null, 2);
            
            const blob = new Blob([debugData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat-debug-${new Date().toISOString()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    });

    // Add visual feedback for debug shortcut
    const debugIndicator = document.createElement('div');
    debugIndicator.style.cssText = `
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        display: none;
        z-index: 1000;
    `;
    debugIndicator.textContent = 'Debug logs downloaded';
    document.body.appendChild(debugIndicator);

    // Show indicator when logs are downloaded
    function showDebugIndicator() {
        debugIndicator.style.display = 'block';
        setTimeout(() => {
            debugIndicator.style.display = 'none';
        }, 2000);
    }

    // Error boundary for unexpected errors
    window.onerror = function(msg, url, lineNo, columnNo, error) {
        logDebugInfo('global_error', {
            message: msg,
            url: url,
            line: lineNo,
            column: columnNo,
            error: error?.stack || 'No stack trace available'
        });
        return false;
    };

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', function(event) {
        logDebugInfo('unhandled_promise_rejection', {
            reason: event.reason?.message || event.reason,
            stack: event.reason?.stack || 'No stack trace available'
        });
    });

    // Performance monitoring
    if (window.PerformanceObserver) {
        const perfObserver = new PerformanceObserver((list) => {
            list.getEntries().forEach((entry) => {
                logDebugInfo('performance', {
                    type: entry.entryType,
                    name: entry.name,
                    duration: entry.duration,
                    startTime: entry.startTime
                });
            });
        });

        perfObserver.observe({ entryTypes: ['navigation', 'resource', 'longtask'] });
    }
});