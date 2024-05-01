from bs4 import BeautifulSoup
from selenium import webdriver
from utils.customLogger import setup_logger
from pathlib import Path
import re
import json
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

logger = setup_logger(Path(__file__).name[:-3])  # set up custom logger

# this is the web scraping program used to capture live tweets, without account cycling implemented
# (meaning it periodically stops working due to twitter restricting the account's access temporarily)

keys = {}
# accessing twitter login from .env file, which is to avoid publishing them on Git (security)
with open(r'..\.env', 'r') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        keys[key]=value

WINDOW_SIZE = "1920,1080"
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
chrome_options.add_argument("--enable-javascript")
chrome_options.binary_location = "C:\Program Files\Google\Chrome\Application\chrome.exe"

driver = webdriver.Chrome(options=chrome_options) #set up webdriver
times=[]
tweets_set = set()

def LoginTwitter(target_url="https://twitter.com/search?q=%23crypto%20min_faves%3A2&src=typed_query&f=live"): #added min 2 likes requirement
    driver.get(target_url)
    username = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]')))
    username.send_keys(keys["TWITTER_EMAIL"])
    username.send_keys(Keys.ENTER)

    try:
        password = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[name="password"]')))
        password.send_keys(keys["TWITTER_PASSWORD"])
        password.send_keys(Keys.ENTER)
    except:
        username = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="on"]')))
        username.send_keys("avmlprojects")
        username.send_keys(Keys.ENTER)
        password = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'input[name="password"]')))
        password.send_keys(keys["TWITTER_PASSWORD"])
        password.send_keys(Keys.ENTER)

def cleanNumber(number):
    number=number.replace(",","")
    if re.search('[a-zA-Z]', number) != None:
        if "K" in number:
            number=number[:-1]
            number=1000*float(number)
        elif "M" in number:
            number=number[:-1]
            number=1000000*float(number)
        elif "B" in number:
            number=number[:-1]
            number=1000000000*float(number)
    return number

class Tweet(object):
    def __init__(self, username, time, text, following, followers):
        self.username = username
        self.time = time
        self.text = text
        self.following = following
        self.followers = followers

class TweetEncoder(json.JSONEncoder):
    def default(self, obj):
            return {"username":obj.username, "time":obj.time, "text":obj.text, "following":obj.following, "followers":obj.followers}
try:
    LoginTwitter()
    time.sleep(3)
    start_time=time.time()
    while len(tweets_set)<200:
        driver.execute_script('window.scrollBy(0, 3000)')
        time.sleep(1)
        resp=driver.page_source
        time.sleep(2)
        soup=BeautifulSoup(markup=resp,features='html.parser')

        all_divs=soup.find_all(name="div",attrs={"data-testid":"cellInnerDiv"}) # get all the tweet cells
        for div in all_divs:
            # get username in format /username
            try:
                username_markup = div.find(role="link", tabindex="-1")
                username=username_markup.attrs["href"]
                logger.info(username)
            except AttributeError:
                continue

            # access profile page, if they don't meet the following criteria, remove their tweet as viable to avoid bots/spam:
            # 1. Have more than 50 followers
            # 2. Have at least 50 % more followers than amount they follow
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get("https://twitter.com"+username)
            time.sleep(3)
            profile_resp = driver.page_source
            time.sleep(1)
            profile_soup = BeautifulSoup(markup=profile_resp, features="html.parser")
            try:
                following=profile_soup.find(name="a",attrs={"href":username+"/following"}).find(name="span").text
                followers=profile_soup.find(name="a",attrs={"href":username+"/verified_followers"}).find(name="span").text
                following=cleanNumber(following)
                followers=cleanNumber(followers)
                logger.info(f"{username} has {followers} followers and {following} following,")
            except AttributeError:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                continue
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            # try to get timestamp, rarely a tweet won't have one (unknown reason)
            try:
                stamp = div.find(name="time")
                timestamp = stamp.attrs["datetime"]
                times.append(stamp.attrs["datetime"])
            except AttributeError:
                logger.warning(f"Tweet {div} didnt have a timestamp")
                times.append("Unavailable")
                stamp="None"

            # get tweet contents
            cleaned_text = ""
            actual_text_list = []
            spans = div.find_all(name="span")
            for span in spans:
                text = span.text
                if "#" in text:
                    continue
                if re.search('[a-zA-Z]', text) == None:
                    continue
                actual_text_list.append(text)
            cleaned_text = ''.join(actual_text_list)
            if int(following)!=0:
                if int(followers)>50 and int(followers)/int(following)>0.5:
                    tweets_set.add(Tweet(username,timestamp,cleaned_text,following,followers))
                else:
                    logger.info(f"{username} is likely a bot, have not included their tweet")
            else:
                if int(followers)>50:
                    tweets_set.add(Tweet(username,timestamp,cleaned_text, following, followers))
                else:
                    logger.info(f"{username} is likely a bot, have not included their tweet")
    for tweet in tweets_set:
        logger.info(f"{tweet.username} tweeted at {tweet.time} \n {tweet.text} ")
    end_time=time.time()
    logger.info(f"Took {end_time-start_time} seconds to get 100 tweets")
finally:
    json_file=open(r"Data/twitter.json", "w")
    json.dump(list(tweets_set),json_file, cls=TweetEncoder,indent=2)
    json_file.close()


