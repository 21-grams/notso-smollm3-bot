// Only function that HTMX can't handle: clipboard copy
function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        navigator.clipboard.writeText(element.textContent).then(() => {
            // Visual feedback using CSS classes
            element.classList.add('copied');
            setTimeout(() => element.classList.remove('copied'), 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    }
}
