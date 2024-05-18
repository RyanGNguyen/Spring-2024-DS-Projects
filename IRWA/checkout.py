import os
try:
    import time
except:
    os.system("python -m pip install time")
    import time
try:
    import sys
except:
    os.system("python -m pip install sys")
    import sys
try:
    import pandas as pd
except:
    os.system("python -m pip install pandas")
    import pandas as pd
try:
    from selenium import webdriver
except:
    os.system("python -m pip install selenium")
    import pandas as pd
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import checkout_bot_bn
import checkout_bot_amazon
try:
    from amazoncaptcha import AmazonCaptcha
except:
    os.system("python -m pip install amazoncaptcha")
    from amazoncaptcha import AmazonCaptcha

# call either BN or Amazon depending on input 
if __name__ == "__main__":
    input = pd.read_csv(sys.argv[1],header=[0])
    store = input.loc[0].at["Store"]
    print(store)
    if "Amazon" in store :
        checkout_bot_amazon.runAmazon()
    elif "BN" in store or "Barnes and Noble" in store:
        checkout_bot_bn.runBN()
    else:
        print("Store not supported")