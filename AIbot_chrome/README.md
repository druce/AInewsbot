Previous attempt to do the same thing as a Chrome extension.

You would think that would be very doable, but even if you inject code into the web pages to wait until the page loads and save the HTML, the web page doesn't have the ability to save itself locally. I guess people could write malicious code to install something nasty you might click on, or just fill the disk.

so you need a service worker, send the page contents to service worker which will save it.

which gets very complicated and for some reason it didn't work for feedly and google news, possibly conflicts with javascript they already have.