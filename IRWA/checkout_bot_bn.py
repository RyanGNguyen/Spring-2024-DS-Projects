import time
import sys
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

class CheckOutBotBN:
    def __init__(self, csv_file):
        self.info = pd.read_csv(csv_file)
        self.driver = webdriver.Chrome()
        self.driver.set_page_load_timeout(120)
        self.driver.get("https://www.barnesandnoble.com/")
        self.wait = WebDriverWait(self.driver, 15)
        self.accept_cookies()

    def accept_cookies(self):
        time.sleep(5)
        self.wait.until(EC.element_to_be_clickable((By.ID,"onetrust-accept-btn-handler")))
        button = self.driver.find_elements(By.ID,"onetrust-accept-btn-handler")
        button[0].click()

    def login(self): 
        time.sleep(10)
        self.driver.get("https://www.barnesandnoble.com/account/")            
        self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "sign-into-account")))
        login = self.driver.find_elements(By.CLASS_NAME, "sign-into-account")
        login[0].click()

        time.sleep(10)
        self.wait.until(EC.visibility_of_element_located((By.XPATH, "//iframe[@title='Sign in or Create an Account']")))
        frame = self.driver.find_elements(By.XPATH, "//iframe[@title='Sign in or Create an Account']")
        self.driver.switch_to.frame(frame[0])

        time.sleep(10)
        self.wait.until(EC.visibility_of_element_located((By.ID, "email")))
        email_input = self.driver.find_elements(By.ID, "email")
        email_input[0].clear()
        email_input[0].send_keys(self.info["Email"][0])
        
        time.sleep(10)
        self.wait.until(EC.visibility_of_element_located((By.ID, "password")))
        pass_input = self.driver.find_elements(By.ID, "password")
        pass_input[0].clear()
        pass_input[0].send_keys(self.info["Password"][0])

        time.sleep(10)
        self.wait.until(EC.element_to_be_clickable((By.XPATH,"//button[@type='submit']")))
        button = self.driver.find_elements(By.XPATH,"//button[@type='submit']")
        button[0].click()


    def add_book_to_cart(self,book):
        link = "https://www.barnesandnoble.com/s/" + book
        self.driver.get(link)
        time.sleep(3)
        add_to_cart_button = self.driver.find_elements(By.CLASS_NAME, "add-to-cart-button btn-addtocart btn-pdp-addtocart btn btn--commerce btn--commerce-non-digital")
        add_to_cart_button[0].click()
       
    def add_product_to_chart(self, link):
        time.sleep(10)
        self.driver.get(link)
        #self.wait.until(EC.element_to_be_clickable((By.XPATH,"//button[@value='ADD TO CART']")))
        #add_to_cart_button = self.driver.find_elements(By.XPATH,"//button[@value='ADD TO CART']")
        #self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "add-to-cart-button btn-addtocart btn-pdp-addtocart btn btn--commerce btn--commerce-non-digital")))
        #add_to_cart_button = self.driver.find_elements(By.CLASS_NAME, "add-to-cart-button btn-addtocart btn-pdp-addtocart btn btn--commerce btn--commerce-non-digital")
        #self.wait.until(EC.element_to_be_clickable((By.ID, "addToCart")))
        #add_to_cart_button = self.driver.find_elements(By.ID, "addToCart")

        #self.wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "common-purchase-btn--container pr-0")))
        add_to_cart_button = self.driver.find_elements(By.CLASS_NAME, "add-to-cart-button btn-addtocart btn-pdp-addtocart btn btn--commerce btn--commerce-non-digital")
        add_to_cart_button[0].click()
        
        #self.wait.until(EC.element_to_be_clickable((By.ID, "continueShopping")))
        #continue_button = self.driver.find_elements(By.ID, "continueShopping")
        #continue_button[0].click()

    def checkout(self):
        self.driver.get("https://www.barnesandnoble.com/checkout/")         # Verify
        time.sleep(1)
        self.driver.find_element(By.CLASS_NAME, "SelectGroupstyled__SelectGroupItemContainer-sc-1iooaif-0")[2].click()
        time.sleep(1)
        self.driver.find_element(By.CLASS_NAME, "ContinueButton__StyledContinue-fh9abp-0")[1].click()

        # this is how you click the final checkout button
        # self.driver.find_elements_by_class_name(
        #     "ContinueButton__StyledContinue-fh9abp-0"
        # )[2].click()

    def __del__(self):
        self.driver.close()


def runBN():
    file = sys.argv[1]
    
    checkout_bot = CheckOutBotBN(file)

    #checkout_bot.login() 
    checkout_bot.add_book_to_cart(checkout_bot.info["Book"][0])
    checkout_bot.add_product_to_chart(
        "https://www.barnesandnoble.com/w/fire-blood-george-r-r-martin/1128905006?ean=9781524796303"
    )
    #checkout_bot.add_product_to_chart(
    #    "https://www.barnesandnoble.com/w/the-ballad-of-songbirds-and-snakes-suzanne-collins/1133952083?ean=9781339016573"
    #)
    #checkout_bot.add_product_to_chart(
    #    "https://www.barnesandnoble.com/w/the-stand-stephen-king/1100631608?ean=9780307947307"
    #)
    
    checkout_bot.checkout()
    time.sleep(20)