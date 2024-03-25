// content-script.js
// some files the first function works, some files the second function works, google and feedly, neither works
// those are the 2 with infinite scroll

window.onload = function() {
    // alert(document.title);
    chrome.runtime.sendMessage({
        action: 'savePageContent',
        pageTitle: document.title + "-onload",
        pageContent: document.documentElement.outerHTML
    });
};

window.addEventListener('load', function() {
    const blob = new Blob([document.documentElement.outerHTML], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    chrome.runtime.sendMessage({
        action: 'savePageContent',
        pageTitle: document.title+"-load",
        pageContent: url
    });
});
