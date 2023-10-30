import requests
import pickle
import browser_cookie3
import selenium.webdriver
import os

cookie_name = "openAIKey"
cookie_fname = "cookies.pcl"

def saveOpenAIKey(value):
    global cookie_name, cookie_fname

    print(f"Saving the value in cookie...")

    s = requests.session()
    s.cookies.set(cookie_name, value)

    #print(f"Session cookies before save: {s.cookies}")

    # Save the cookies to file:
    #with open(cookie_fname, 'wb') as f:
    #    pickle.dump(s.cookies, f)

    # Chrome browser
    try:
        driver = selenium.webdriver.Chrome()
        driver.get("https://huggingface.co")
        driver.add_cookie({cookie_name: value})
    except Exception as e:
        print(f"Exception: {e}")

def loadOpenAIKey():
    global cookie_name, cookie_fname

    openAIkey = None
    
    print(f"Loading the value from cookie...")
    s = requests.session()

    #try:
    #    if os.path.exists(cookie_fname):
    #        with open(cookie_fname, 'rb') as f:
    #            s.cookies.update(pickle.load(f))
    #except Exception as e:
    #    print(f"Exception: {f}")

    print(f"Saved cokies: {s.cookies}")

    openAIkey = s.cookies.get(cookie_name)
    print(f"Server cookie: {openAIkey!=None}")
    if openAIkey == None:
        try:
            driver = selenium.webdriver.Chrome()
            driver.get("https://huggingface.co")
            print("Cookies from Chrome:")
            for cookie in driver.get_cookies():
                print(cookie)
                if cookie_name in cookie:
                    print("Found open ai key!")
                    openAIkey = cookie[cookie_name]
        except Exception as e:
            print(f"Exception: {e}")

    return openAIkey