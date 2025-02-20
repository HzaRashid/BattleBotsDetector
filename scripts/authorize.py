from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from argparse import ArgumentParser
from time import sleep

def main():
    parser = ArgumentParser()
    parser.add_argument('--key', type=str)
    parser.add_argument('--api-docs', type=str)
    parser.add_argument('--browser', type=str, default='chrome')
    args = parser.parse_args()
    # Set up WebDriver (Make sure to download ChromeDriver or use another browser)
    driver = getattr(webdriver,
                     {'chrome': 'Chrome',
                      'safari': 'Safari',
                      'firefox': 'Firefox',
                      'edge': 'Edge'
                      }['chrome' if args.browser=='None' else args.browser.lower()]
                    )()


    # Open the website
    driver.get(args.api_docs)

    # Find the "Authorize" button and click it
    # 🔹 Wait until the button is clickable
    authorize_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.authorize.unlocked"))
    )
    authorize_button.click()

    # Find the input field and enter the token
    token_value = args.key
    # 🔹 Wait until the input field is present
    input_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "auth-bearer-value"))
    )
    input_field.send_keys(token_value)
    sleep(1)

    # Find the "Authorize" button and click it
    authorize_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.modal-btn.auth.authorize.button"))
    )
    authorize_button.click()

    sleep(1)

    # Close the browser (optional)
    driver.quit()

if __name__ == "__main__":
    main()