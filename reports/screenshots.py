"""Скриншоты приложения для отчёта (Playwright + Microsoft Edge)."""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
URL = "http://localhost:8599"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1750}, device_scale_factor=1.5)
    for _ in range(30):
        try:
            page.goto(URL, timeout=5000)
            break
        except Exception:
            time.sleep(2)
    page.wait_for_selector("text=Рассчитать риск", timeout=60000)
    time.sleep(2)
    page.screenshot(path=str(OUT / "fig10_main.png"), full_page=True)

    page.get_by_role("button", name="Рассчитать риск").click()
    page.wait_for_selector("text=Вероятность заболевания", timeout=30000)
    time.sleep(2)
    page.screenshot(path=str(OUT / "fig11_result.png"), full_page=True)

    page.get_by_role("tab", name="Загрузка CSV").click()
    time.sleep(1)
    page.locator("input[type=file]").set_input_files(str(ROOT / "data" / "sample_patients.csv"))
    page.wait_for_selector("text=Обработано пациентов", timeout=30000)
    time.sleep(2)
    page.screenshot(path=str(OUT / "fig12_csv.png"), full_page=True)
    browser.close()
print("screenshots saved")
