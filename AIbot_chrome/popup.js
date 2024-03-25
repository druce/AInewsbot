// popup.js

document.getElementById('start-button').addEventListener('click', () => {
    // Send a message to the background script to start the process
    chrome.runtime.sendMessage({ action: 'startFetchingNews' });

    // Optionally, close the popup after the button is clicked
    window.close();
});
