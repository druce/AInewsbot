// background.js
console.log("starting background.js");

// Function to sanitize and format the filename for the HTML file
function formatFileName(title) {
    const date = new Date();
    const dateString = date.toISOString().replace(/[:.]/g, '-');
    const sanitizedTitle = title.replace(/[^a-zA-Z0-9]/g, '_');
    return `${sanitizedTitle}-${dateString}.html`;
}

// Function to save the HTML content of a page
// function saveHTMLContent(tabId, content, title) {
//     const fileName = formatFileName(title);
//     const blob = new Blob([content], { type: 'text/html' });
//     const url = URL.createObjectURL(blob);

//     chrome.downloads.download({
//         url: url,
//         filename: fileName,
//         saveAs: false
//     }, () => {
//         URL.revokeObjectURL(url); // Free up memory
//         chrome.tabs.remove(tabId); // Close the tab after saving
//     });
// }

// Function to open a URL in a new tab and save its content
function processBookmark(bookmark) {
    chrome.tabs.create({ url: bookmark.url, active: false }, (tab) => {
        const tabId = tab.id;

        // Wait for the page to fully load before capturing the content
        // chrome.tabs.onUpdated.addListener(function listener(tabId, changeInfo) {
        //     if (changeInfo.status === 'complete') {
        //         // Tell the content script to send the page content
        //         chrome.tabs.sendMessage(tabId, { action: 'fetchPageContent' });

        //         // Remove this listener after the task is complete to avoid memory leak
        //         chrome.tabs.onUpdated.removeListener(listener);
        //     }
        // });

        // Listen for the content script to send back the HTML content
        // chrome.runtime.onMessage.addListener(function listener(message, sender, sendResponse) {
        //     if (message.action === 'saveHTML' && sender.tab.id === tabId) {
        //         saveHTMLContent(tabId, message.content, message.title);

        //         // Remove this listener to avoid memory leak
        //         chrome.runtime.onMessage.removeListener(listener);
        //     }
        // });
    });
}

// Function to process all bookmarks in the "Tech news" folder
function processBookmarks(bookmarks) {
    bookmarks.forEach((bookmark) => {
        if (bookmark.children) {
            processBookmarks(bookmark.children); // Recurse into folders
        } else {
            processBookmark(bookmark);
        }
    });
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    console.log("clicked");
    if (request.action === 'startFetchingNews') {
        chrome.bookmarks.search({ title: 'Tech news' }, (results) => {
            results.forEach((result) => {
                chrome.bookmarks.getSubTree(result.id, (tree) => {
                    processBookmarks(tree);
                });
            });
        });
    }
    else if (request.action === 'savePageContent') {
        console.log("saving " + request.pageTitle);

        fileName = formatFileName(request.pageTitle);
        chrome.downloads.download({
            url: request.pageContent,
            filename: fileName,
            saveAs: false  // prompts the user to select a save location; set to false to save directly
        // }, (downloadId) => {
        //     // This callback is optional and can be used to handle the download id
        //     console.log(`Download started with ID: ${downloadId}`);
        });

        // Cleanup the blob URL after the download starts
        // chrome.downloads.onChanged.addListener((delta) => {
        // if (delta.id === downloadId && delta.state && delta.state.current === "in_progress") {
        //     URL.revokeObjectURL(url);
        // }
        // });

        // sendResponse({ status: 'File is being saved' });
    }

});

// // When the extension's button is clicked, start the process
// chrome.action.onClicked.addListener(() => {
//     console.log("clicked");
//     chrome.bookmarks.search({ title: 'Tech news' }, (results) => {
//         results.forEach((result) => {
//             chrome.bookmarks.getSubTree(result.id, (tree) => {
//                 processBookmarks(tree);
//             });
//         });
//     });
// });
