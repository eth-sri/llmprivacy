from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium_stealth import stealth
from tqdm import tqdm
import time


from src.configs import ModelConfig
from src.models import BaseModel
from src.prompts import Prompt
from src.utils.limiter import RateLimiter
from typing import List, Tuple, Iterator

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")

# options.add_argument("--headless")

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)


def clean_string(input: str) -> str:
    """Removes all non BMP characters from a string

    Args:
        input (str): string to be cleaned

    Returns:
        str: cleaned string
    """
    return "".join(c for c in input if c <= "\uFFFF")


# def login_to_poe(username, driver):
#     driver.get(url)  # navigate to the page

#     username = q = driver.find_element(
#         By.CSS_SELECTOR, "input[class^='EmailInput_emailInput_']"
#     )
#     username.send_keys("username")
#     username.send_keys(Keys.RETURN)


def switch_to_bot(bot, driver: WebDriver):
    bot_menu = driver.find_element(
        By.CSS_SELECTOR, "[class^='ChatPageBotSwitcher_navigationIcon__']"
    )
    time.sleep(1)
    bot_menu.click()
    time.sleep(2)
    select_bot = driver.find_elements(By.CSS_SELECTOR, "[href='/" + bot + "']")
    select_bot[-1].click()


def send_prompt(prompt: str, driver: WebDriver):
    # Clear chat
    input_area = driver.find_element(
        By.CSS_SELECTOR, "[class^='GrowingTextArea_textArea']"
    )

    clear_button = driver.find_element(
        By.XPATH, "//*[@id='__next']/div[1]/main/div/div/div/footer/div/button"
    )

    clear_button.click()
    time.sleep(1)

    split_prompt = prompt.split("\n")
    for line in split_prompt:
        if len(line) > 0:
            input_area.send_keys(clean_string(line))

        ActionChains(driver).key_down(Keys.SHIFT).key_down(Keys.ENTER).key_up(
            Keys.SHIFT
        ).key_up(Keys.ENTER).perform()

    time.sleep(0.25)
    input_area.send_keys(Keys.ENTER)

    output_wrapper = driver.find_elements(
        By.CSS_SELECTOR, "[class^='Message_botMessageBubble__']"
    )
    ctr = 0

    last_message = ""
    while ctr < 30:
        feedback = driver.find_elements(
            By.CSS_SELECTOR,
            "section[class^='ChatMessageFeedbackButtons_feedbackButtonsContainer__']",
        )
        if len(feedback) > 0:
            break
        time.sleep(2)
        ctr += 1

    output_wrapper = driver.find_elements(
        By.CSS_SELECTOR, "[class^='Message_botMessageBubble__']"
    )
    try:
        last_message = output_wrapper[-1].text
    except Exception:
        last_message = ""

    if len(last_message) < 10:
        print("No output")
        return ""

    return last_message


class PoeModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        chrome_options = Options()
        # chrome_options.add_argument("user-data-dir=selenium")
        driver = webdriver.Chrome(options=chrome_options)
        stealth(
            driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
        )

        url = "https://www.poe.com"
        driver.get(url)  # navigate to the page

        is_logged_in = False
        while not is_logged_in:
            time.sleep(1)
            bot_sel_elements = driver.find_elements(
                By.CSS_SELECTOR, "[class^='ChatPageBotSwitcher_navigationIcon__']"
            )
            is_logged_in = len(bot_sel_elements) > 0

        switch_to_bot(config.name, driver)
        time.sleep(2)
        self.driver = driver

    def predict(self, input: Prompt, **kwargs) -> str:
        response = send_prompt(
            self.apply_model_template(input.get_prompt()), self.driver
        )
        return response

    def predict_string(self, input: str, **kwargs) -> str:
        response = send_prompt(input, self.driver)

        return response

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        ids_to_do = list(range(len(inputs)))
        retry_ctr = 0

        rl = RateLimiter(1, 20)

        ids_to_do = list(range(len(inputs)))

        with tqdm(total=len(ids_to_do)) as pbar:
            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                id = ids_to_do[0]
                if not rl.record():
                    print("Rate limit exceeded, sleeping for 5 seconds")
                    time.sleep(5)
                    continue

                orig = inputs[id]
                answer = self.predict(inputs[id])
                if answer == "":
                    retry_ctr += 1
                    continue
                yield (orig, answer)
                ids_to_do.remove(id)
                pbar.n = len(inputs) - len(ids_to_do)
                pbar.refresh()
