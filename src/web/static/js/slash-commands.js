// Slash Commands Module
(function() {
    'use strict';
    
    // Available slash commands
    const commands = [
        // Chat Commands
        {
            category: 'Chat',
            name: 'clear',
            description: 'Clear the chat history',
            icon: '🗑️',
            shortcut: '/clear',
            action: () => clearChat()
        },
        {
            category: 'Chat',
            name: 'reset',
            description: 'Reset the conversation context',
            icon: '🔄',
            shortcut: '/reset',
            action: () => resetContext()
        },
        {
            category: 'Chat',
            name: 'export',
            description: 'Export chat history to file',
            icon: '💾',
            shortcut: '/export',
            action: () => exportChat()
        },
        
        // Model Commands
        {
            category: 'Model',
            name: 'thinking',
            description: 'Toggle thinking/reasoning mode',
            icon: '🤔',
            shortcut: '/thinking',
            action: () => toggleThinking()
        },
        {
            category: 'Model',
            name: 'temperature',
            description: 'Adjust response creativity (0.0-1.0)',
            icon: '🌡️',
            shortcut: '/temp',
            action: (value) => setTemperature(value)
        },
        {
            category: 'Model',
            name: 'model',
            description: 'Switch model or view current model',
            icon: '🤖',
            shortcut: '/model',
            action: () => showModelInfo()
        },
        
        // Utility Commands
        {
            category: 'Utility',
            name: 'help',
            description: 'Show all available commands',
            icon: '❓',
            shortcut: '/help',
            action: () => showHelp()
        },
        {
            category: 'Utility',
            name: 'status',
            description: 'Show system status and metrics',
            icon: '📊',
            shortcut: '/status',
            action: () => showStatus()
        },
        {
            category: 'Utility',
            name: 'theme',
            description: 'Toggle dark/light theme',
            icon: '🎨',
            shortcut: '/theme',
            action: () => toggleTheme()
        },
        {
            category: 'Utility',
            name: 'settings',
            description: 'Open settings panel',
            icon: '⚙️',
            shortcut: '/settings',
            action: () => openSettings()
        },
        
        // Quick Actions
        {
            category: 'Quick Actions',
            name: 'quote',
            description: 'Quote scripture passage (e.g., john 1:1-14)',
            icon: '📖',
            shortcut: '/quote',
            action: () => {
                const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
                // Just send the command as a message to the server
                input.value = '/quote john 1:1-14';
                input.form.requestSubmit();
            }
        },
        {
            category: 'Quick Actions',
            name: 'summarize',
            description: 'Summarize the conversation',
            icon: '📝',
            shortcut: '/summarize',
            action: () => summarizeChat()
        },
        {
            category: 'Quick Actions',
            name: 'continue',
            description: 'Continue the last response',
            icon: '➡️',
            shortcut: '/continue',
            action: () => continueResponse()
        },
        {
            category: 'Quick Actions',
            name: 'regenerate',
            description: 'Regenerate the last response',
            icon: '🔁',
            shortcut: '/regenerate',
            action: () => regenerateResponse()
        }
    ];
    
    let selectedIndex = 0;
    let filteredCommands = [...commands];
    let menuActive = false;
    
    // Initialize slash commands
    function init() {
        const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
        if (!input) return;
        
        // Find the input wrapper (parent div with position: relative)
        const inputWrapper = input.closest('.input-wrapper') || input.parentElement;
        
        // Create command menu
        const menu = createCommandMenu();
        
        // Ensure wrapper has relative positioning
        if (inputWrapper) {
            inputWrapper.style.position = 'relative';
            inputWrapper.appendChild(menu);
        } else {
            // Fallback: wrap input in a div
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            wrapper.className = 'input-wrapper';
            input.parentNode.insertBefore(wrapper, input);
            wrapper.appendChild(input);
            wrapper.appendChild(menu);
        }
        
        // Listen for input changes
        input.addEventListener('input', handleInput);
        input.addEventListener('keydown', handleKeyDown);
        
        // Close menu on outside click
        document.addEventListener('click', (e) => {
            if (!menu.contains(e.target) && e.target !== input) {
                hideMenu();
            }
        });
    }
    
    // Create the command menu DOM
    function createCommandMenu() {
        const menu = document.createElement('div');
        menu.className = 'slash-command-menu';
        menu.id = 'slash-command-menu';
        
        // Add search input
        const search = document.createElement('input');
        search.type = 'text';
        search.className = 'slash-command-search';
        search.placeholder = 'Search commands...';
        search.addEventListener('input', (e) => filterCommands(e.target.value));
        
        // Add command list
        const list = document.createElement('ul');
        list.className = 'slash-command-list';
        list.id = 'slash-command-list';
        
        menu.appendChild(search);
        menu.appendChild(list);
        
        return menu;
    }
    
    // Handle input changes
    function handleInput(e) {
        const value = e.target.value;
        const lastChar = value[value.length - 1];
        const cursorPos = e.target.selectionStart;
        
        // Check if user typed '/' at the beginning or after a space
        if (lastChar === '/' && (cursorPos === 1 || value[cursorPos - 2] === ' ')) {
            showMenu();
        } else if (menuActive) {
            // Check if still in command mode
            const commandMatch = value.match(/\/(\w*)?$/);
            if (commandMatch) {
                filterCommands(commandMatch[1] || '');
            } else {
                hideMenu();
            }
        }
    }
    
    // Handle keyboard navigation
    function handleKeyDown(e) {
        if (!menuActive) return;
        
        switch(e.key) {
            case 'ArrowDown':
                e.preventDefault();
                selectedIndex = Math.min(selectedIndex + 1, filteredCommands.length - 1);
                updateSelection();
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                selectedIndex = Math.max(selectedIndex - 1, 0);
                updateSelection();
                break;
                
            case 'Enter':
                if (filteredCommands[selectedIndex]) {
                    e.preventDefault();
                    executeCommand(filteredCommands[selectedIndex]);
                }
                break;
                
            case 'Tab':
                if (filteredCommands[selectedIndex]) {
                    e.preventDefault();
                    autocompleteCommand(filteredCommands[selectedIndex]);
                }
                break;
                
            case 'Escape':
                e.preventDefault();
                hideMenu();
                break;
        }
    }
    
    // Show command menu
    function showMenu() {
        const menu = document.getElementById('slash-command-menu');
        menu.classList.add('active');
        menuActive = true;
        selectedIndex = 0;
        renderCommands();
        
        // Focus search if exists
        const search = menu.querySelector('.slash-command-search');
        if (search) {
            setTimeout(() => search.focus(), 50);
        }
    }
    
    // Hide command menu
    function hideMenu() {
        const menu = document.getElementById('slash-command-menu');
        menu.classList.remove('active');
        menuActive = false;
    }
    
    // Filter commands based on search
    function filterCommands(query) {
        if (!query) {
            filteredCommands = [...commands];
        } else {
            filteredCommands = commands.filter(cmd => 
                cmd.name.toLowerCase().includes(query.toLowerCase()) ||
                cmd.description.toLowerCase().includes(query.toLowerCase()) ||
                cmd.category.toLowerCase().includes(query.toLowerCase())
            );
        }
        selectedIndex = 0;
        renderCommands();
    }
    
    // Render filtered commands
    function renderCommands() {
        const list = document.getElementById('slash-command-list');
        list.innerHTML = '';
        
        if (filteredCommands.length === 0) {
            list.innerHTML = '<div class="slash-command-empty">No commands found</div>';
            return;
        }
        
        let currentCategory = '';
        filteredCommands.forEach((cmd, index) => {
            // Add category header if new category
            if (cmd.category !== currentCategory) {
                currentCategory = cmd.category;
                const categoryEl = document.createElement('div');
                categoryEl.className = 'slash-command-category';
                categoryEl.textContent = currentCategory;
                list.appendChild(categoryEl);
            }
            
            // Add command item
            const item = document.createElement('li');
            item.className = 'slash-command-item';
            if (index === selectedIndex) {
                item.classList.add('selected');
            }
            
            item.innerHTML = `
                <div class="slash-command-icon">${cmd.icon}</div>
                <div class="slash-command-content">
                    <div class="slash-command-name">${cmd.name}</div>
                    <div class="slash-command-description">${cmd.description}</div>
                </div>
                <div class="slash-command-shortcut">${cmd.shortcut}</div>
            `;
            
            item.addEventListener('click', () => executeCommand(cmd));
            item.addEventListener('mouseenter', () => {
                selectedIndex = index;
                updateSelection();
            });
            
            list.appendChild(item);
        });
    }
    
    // Update visual selection
    function updateSelection() {
        const items = document.querySelectorAll('.slash-command-item');
        items.forEach((item, index) => {
            if (index === selectedIndex) {
                item.classList.add('selected');
                item.scrollIntoView({ block: 'nearest' });
            } else {
                item.classList.remove('selected');
            }
        });
    }
    
    // Execute selected command
    function executeCommand(cmd) {
        const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
        
        // Clear the command from input
        input.value = input.value.replace(/\/\w*$/, '');
        
        // Execute command action
        cmd.action();
        
        // Hide menu
        hideMenu();
        
        // Focus back to input
        input.focus();
    }
    
    // Autocomplete command in input
    function autocompleteCommand(cmd) {
        const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
        input.value = input.value.replace(/\/\w*$/, cmd.shortcut + ' ');
        hideMenu();
        input.focus();
    }
    
    // Command Actions
    function clearChat() {
        const messages = document.getElementById('chat-messages');
        if (messages && confirm('Clear all chat history?')) {
            messages.innerHTML = '';
            console.log('Chat cleared');
        }
    }
    
    function resetContext() {
        // Send reset request to server
        fetch('/api/reset-context', { method: 'POST' })
            .then(() => console.log('Context reset'))
            .catch(err => console.error('Reset failed:', err));
    }
    
    function exportChat() {
        const messages = document.querySelectorAll('.message');
        let content = 'Chat Export\n' + '='.repeat(50) + '\n\n';
        
        messages.forEach(msg => {
            const role = msg.classList.contains('user') ? 'User' : 'Assistant';
            const text = msg.querySelector('.message-bubble').textContent;
            content += `${role}: ${text}\n\n`;
        });
        
        // Download as file
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().toISOString().slice(0,10)}.txt`;
        a.click();
    }
    
    function toggleThinking() {
        const checkbox = document.getElementById('thinking-mode');
        if (checkbox) {
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        }
    }
    
    function setTemperature(value) {
        const temp = prompt('Set temperature (0.0 - 1.0):', '0.7');
        if (temp && !isNaN(temp)) {
            fetch('/api/set-temperature', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ temperature: parseFloat(temp) })
            });
        }
    }
    
    function showModelInfo() {
        fetch('/api/model-info')
            .then(res => res.json())
            .then(data => {
                alert(`Model: ${data.model}\nStatus: ${data.status}`);
            });
    }
    
    function showHelp() {
        let helpText = 'Available Commands:\n\n';
        commands.forEach(cmd => {
            helpText += `${cmd.shortcut} - ${cmd.description}\n`;
        });
        alert(helpText);
    }
    
    function showStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                alert(`System Status:\n${JSON.stringify(data, null, 2)}`);
            });
    }
    
    function toggleTheme() {
        document.body.classList.toggle('dark-theme');
        localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
    }
    
    function openSettings() {
        console.log('Opening settings panel...');
        // Could open a modal or redirect to settings page
    }
    
    function summarizeChat() {
        const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
        input.value = 'Please summarize our conversation so far.';
        input.form.requestSubmit();
    }
    
    function continueResponse() {
        const input = document.getElementById('message-input') || document.querySelector('textarea[name="message"]');
        input.value = 'Please continue your last response.';
        input.form.requestSubmit();
    }
    
    function regenerateResponse() {
        // Find last assistant message and regenerate
        console.log('Regenerating last response...');
        // Implementation would depend on your backend
    }
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
    
    // Export for use in other modules if needed
    window.SlashCommands = {
        init,
        showMenu,
        hideMenu,
        addCommand: (cmd) => commands.push(cmd),
        executeCommand
    };
})();
