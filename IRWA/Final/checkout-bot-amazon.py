import time
import sys
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from amazoncaptcha import AmazonCaptcha

class CheckOutBotAmazon:
    def __init__(self, csv_file):
        self.info = pd.read_csv(csv_file,header=[0])
        self.driver = webdriver.Chrome()
        self.driver.get("https://www.amazon.com")
        time.sleep(5)
        # self.accept_cookies()

    def accept_cookies(self):
        button = self.driver.find_elements(By.ID,"privacy-layer-accept-all-button")
        if len(button)==0:
             button = self.driver.find_elements(By.ID,"onetrust-accept-btn-handler")
        button[0].click()

    def login(self): 
        self.driver.get("https://www.amazon.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.com%2F%3Fref_%3Dnav_signin&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=usflex&openid.mode=checkid_setup&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0")            # Verify 
        time.sleep(1)
        captcha = self.driver.find_elements(By.XPATH,"//img")
        if(len(captcha) != 0):
            captcha = AmazonCaptcha.fromlink(captcha[0].get_attribute("src"))
            solution = captcha.solve()
            print(solution)
            time.sleep(2)
            cap_in = self.driver.find_element(By.ID,"captchacharacters")
            cap_in.send_keys(solution)
            self.driver.find_element(By.CLASS_NAME,"a-button-text").click()
        time.sleep(9)
        # # login = self.driver.find_element(By.CLASS_NAME,"sign-into-account")
        # login = self.driver.find_elements(By.CLASS_NAME,"sign-into-account")
        # login[0].click()

        # element_to_hover_over = self.driver.find_element(By.ID,"accountDropdown")

        # hover = ActionChains(self.driver).move_to_element(element_to_hover_over)
        # hover.perform()
        # login = self.driver.find_element(By.CSS_SELECTOR,"a[href='/#']")

        # TODO: doesn't actually click 
        email_input = self.driver.find_element(By.ID,"ap_email")
        email_input.clear()
        print(self.info.to_string())
        print(self.info.columns.tolist())
        print(self.info["Email"][0])
        email_input.send_keys(self.info["Email"][0])
        time.sleep(3)
        cont = self.driver.find_element(By.ID,"continue")
        cont.click()
        time.sleep(3)
        pass_input = self.driver.find_element(By.ID,"ap_password")
        pass_input.clear()
        pass_input.send_keys(self.info["Password"][0])

        self.driver.find_element(By.ID,"signInSubmit").click()
        time.sleep(3)
        captcha = self.driver.find_elements(By.XPATH,"//img")
        if(len(captcha) != 0):
            captcha = AmazonCaptcha.fromlink(captcha[0].get_attribute("src"))
            solution = captcha.solve()
            print(solution)
            time.sleep(2)
            cap_in = self.driver.find_element(By.ID,"captchacharacters")
            cap_in.send_keys(solution)
            self.driver.find_element(By.CLASS_NAME,"a-button-text").click()

    def add_product_to_chart(self, link):
        self.driver.get(link)
        time.sleep(1)
        add_to_cart_button = self.driver.find_element(By.ID,
            'add-to-cart-button'
        )
        time.sleep(2)
        add_to_cart_button.click()

    def checkout(self):
        self.driver.get("https://www.amazon.com/gp/cart/view.html?ref_=nav_cart")         # Verify
        time.sleep(1)
        self.driver.find_element(By.NAME,
            "proceedToRetailCheckout"
        ).click()
        time.sleep(2)
        line1_in = self.driver.find_elements(By.ID,"address-ui-widgets-enterAddressLine1")
        if(len(line1_in)==0):
             checkout_bot.login() 
             time.sleep(5)
             line1_in = self.driver.find_elements(By.ID,"address-ui-widgets-enterAddressLine1")
        line1_in[0].send_keys(self.info["Street Address"][0])
        city_in = self.driver.find_element(By.ID,"address-ui-widgets-enterAddressCity")
        city_in.send_keys(self.info["City"][0])
        zip_in = self.driver.find_element(By.ID,"address-ui-widgets-enterAddressPostalCode")
        zip_in.send_keys(self.info["Zip Code"][0])

        # this is how you click the final checkout button
        # self.driver.find_elements_by_class_name(
        #     "ContinueButton__StyledContinue-fh9abp-0"
        # )[2].click()

    def __del__(self):
        self.driver.close()


if __name__ == "__main__":
    file = sys.argv[1]
    
    checkout_bot = CheckOutBotAmazon(file)

    checkout_bot.login() 
    
    checkout_bot.add_product_to_chart(
       "https://www.amazon.com/Natural-Language-Processing-Corpora-Technology/dp/9048153492/ref=sr_1_2?crid=3GM1C0APISJAR&dib=eyJ2IjoiMSJ9.3p5Ubg_svumNAGEJSHAHIQ-KYG8inqM-EMnZaZqvGg2qn1jA_8P6Vw4swSG_5CZjpp_UgsQlU_GoaXUc5rArVUZipaUpsjMLRxmEXMiZPMM.KQFzPlnqXsWIS47la16wMDAxBOsjwEFgSAz0WXZW5VE&dib_tag=se&keywords=yarowsky&qid=1714164167&s=books&sprefix=yarowsky%2Cstripbooks%2C86&sr=1-2"
    )
    checkout_bot.add_product_to_chart(
        "https://www.amazon.com/Rolling-Stone-Taylor-Swift-Editors/dp/1547865628/ref=sr_1_9?crid=13OPHH61LY90G&dib=eyJ2IjoiMSJ9.dYdWVjkM-RXMxe7Kt_LnRDriS5_-EE2xcwI0MZo81RjFE-F06Q65Aah6CF4Ez6puANGaRkA-GtJdtNCP8_Kb3r_6urgui4tzBbhyACpN9LJkgLsyRL8y-hA3WqoLzvoMePVdS8Cj0kOiKQCqBqOjy32vhQ-tR1NiVvR0j--WijGaoA1_jL7n2oa8i8Co3SRcRAwTA6-sIEUsZIaRT30N9n1DijlePOUvCuBRJ1Wgh4c._Rgw0fmNgNVlOxVJ0XNgIFXcxSan7J3JjVgdgBlkxxs&dib_tag=se&keywords=taylor+swift&qid=1714164242&s=books&sprefix=taylor+swif%2Cstripbooks%2C86&sr=1-9"
    )
    
    checkout_bot.checkout()
    time.sleep(20)
