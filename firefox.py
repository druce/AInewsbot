
"""
Run an interactive playwright session to log in using a persistent browser profile.
Test profile, add logins and cookies
"""
import time

from playwright.sync_api import sync_playwright

from ainewsbot.config import FIREFOX_PROFILE_PATH

with sync_playwright() as p:
    # ⬇️ one context *per* run – but data stays on disk
    context = p.firefox.launch_persistent_context(
        user_data_dir=FIREFOX_PROFILE_PATH,   # <— key line
        headless=False,                   # watch it the first time so you can log in
        viewport={"width": 1280, "height": 800},
        accept_downloads=True,
    )
    page = context.new_page()
    page.goto("https://www.nytimes.com/")
    # 👉 First run: click “Log in” and complete the flow manually.
    # 👉 Later runs: cookies are already there, you’ll be signed in automatically.
    time.sleep(3600)
    context.close()
