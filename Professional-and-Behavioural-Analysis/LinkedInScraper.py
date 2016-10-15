from bs4 import BeautifulSoup
from selenium import webdriver
import json
from selenium.webdriver.common.keys import Keys
import re
import unittest
import csv
import time
import sys
import getpass
def fix_unicode(data):
	    if isinstance(data, bytes):
	        return data.encode('utf-8')
	    elif isinstance(data, dict):
	        data = dict((fix_unicode(k), fix_unicode(data[k])) for k in data);
	    elif isinstance(data, list):
	        for i in xrange(0, len(data)):
	            data[i] = fix_unicode(data[i])
	    return data 
def refine(t):
    t=fix_unicode(t)
    tt= t.strip(' \n\t')
    tt=re.sub('\s+',' ',tt)
    #tt=t.translate(None,'\t\n')
    tt=tt.replace('\xc2\xa0','')
    return(tt.replace(',',''))

print("Please enter your user name for Linkedin")
usr=input()
Profile_Url="https://www.linkedin.com/in/"+usr
#driver=webdriver.PhantomJS()
chromedriver='/home/subir_sbr/Desktop/chromedriver'
driver = webdriver.Chrome(chromedriver)
driver.get(Profile_Url)
Profile_Completeness_Score=0
print("Please wait.. Reviewing your profile details..")
###################################################################################################
try:
	Name=driver.find_element_by_xpath("//h1[@id='name'][@class='fn']").text
except:
	Name=''
try:
	Position=driver.find_element_by_css_selector('p.headline.title').text
except:
	Position=''
try:
	Previous_Position=driver.find_element_by_xpath("//table[@class='extra-info']/tbody/tr[2]/td").text
except:
	Previous_Position=''
try:
	Summary=driver.find_element_by_xpath("//div[@class='description']").text
except:
	Summary=''
try:
	Education=driver.find_element_by_xpath("//table[@class='extra-info']/tbody/tr[3]/td").text
except:
	Education=''

############################# publications ##########################################################
try:
	Number_of_Publications=[]
	elements=driver.find_elements_by_xpath("//section[@id='publications']/ul/li")
	for elm in elements:
		Number_of_Publications.append(refine(elm.text))
except:
	Number_of_Publications=''

Publication_House=[]
try:
	elments=driver.find_elements_by_xpath("//section[@id='publications']/ul/li/header/h5")
	for elm in elments:
		Publication_House.append(refine(elm.text))
except:
	Publication_House=''

try:
	Paper_Title=[]
	elments=driver.find_elements_by_xpath("//section[@id='publications']/ul/li/header/h4")
	for elm in elments:
		Paper_Title.append(refine(elm.text))
except:
	Paper_Title=''
try:
	Authors=[]
	elments=driver.find_elements_by_xpath("//section[@id='publications']/ul/li/dl/dd")
	for elm in elments:
		Authors.append(refine(elm.text))
except:
	Authors=''

try:
	Publications={}
	for i in range(len(Publication_House)):
		myDict={}
		myDict['Publication House']=Publication_House[i]
		myDict['Papaer Title ']=Paper_Title[i]
		myDict['Authors']=Authors[i]
		Publications[i]=dict(myDict)
except:
	Publications={}


############################ SKILLS ######################################################
Skills=[]
try:
	elements=driver.find_elements_by_xpath("//section[@id='skills']/ul/li//span[@class='wrap']")
	for elm in elements:
		if elm.text is not None: 
			Skills.append(refine(elm.get_attribute('textContent')))
		else: pass
except:
	Skills=[]
################################## experience ##############################################
try:
	Position=[]
	elements=driver.find_elements_by_xpath("//section[@id='experience']/ul/li/header/h4")
	for elm in elements:
		if elm.text is not None: 
			Position.append(refine(elm.text))
		else: pass
	
except:
	Position=[]

try:
	Company=[]
	elements=driver.find_elements_by_xpath("//section[@id='experience']/ul/li/header/h5")
	for elm in elements:
		if elm.text is not None: 
			Company.append(refine(elm.text))
		else: pass
except:
	Company=[]


try:
	Experience={}
	for i in range(len(Position)):
		myDict={}
		myDict['Position']=Position[i]
		myDict['Company Name']=Company[i]
		Experience[i]=dict(myDict)
except:
	Experience={}
#################################### CERTIFICATIONS #######################################
Certifications=[]
try:
	elements=driver.find_elements_by_xpath("//section[@id='certifications']/ul/li/header/h4")
	for elm in elements:
		if elm.text is not None: 
			Certifications.append(refine(elm.get_attribute('textContent')))
		else: pass
except:
	Certifications=[]

#################################### ORGANISATIONS ########################################
Organisations=[]
try:
	elements=driver.find_elements_by_xpath("//section[@id='organizations']/ul/li/header/h4")
	for elm in elements:
		if elm.text is not None: 
			Organisations.append(refine(elm.text))
		else: pass
except:
	Organisations=[]
#################################### GROUPS ###############################################
Groups=[]
try:
	elements=driver.find_elements_by_xpath("//section[@id='groups']/ul/li/h4")
	for elm in elements:
		if elm.text is not None: 
			Groups.append(refine(elm.get_attribute('textContent')))
		else: pass
except:
	Groups=[]

########################### COURSES ########################################################
try:
	Courses=[]
	elments=driver.find_elements_by_xpath("//section[@id='courses']/ul/li/div/ul/li")
	for elm in elments:
		Courses.append(refine(elm.text))
	
except:
	Courses=[]

########################## PROJECTS ################################################
	
try:
	Number_of_Projects=driver.find_elements_by_xpath("//section[@id='projects']/ul/li")
except:
	Number_of_Projects=''

try:
	Headers=[]
	elments=driver.find_elements_by_xpath("//section[@id='projects']/ul/li/header/h4")
	for elm in elments:
		Headers.append(refine(elm.text))
except:
	Headers=[]

try:
	Descripion=[]
	elments=driver.find_elements_by_xpath("//section[@id='projects']/ul/li/p")
	for elm in elments:
		Descripion.append(refine(elm.text))
except:
	Descripion=[]

try:
	Contributors=[]
	elments=driver.find_elements_by_xpath("//section[@id='projects']/ul/li/dl/dd")
	for elm in elments:
		Contributors.append(refine(elm.text))
except:
	Contributors=[]


Project={}
for i in range(len(Headers)):
	myDict={}
	myDict['Topic']=Headers[i]
	myDict['Description']=Descripion[i]
	myDict['Contributors']=Contributors[i]
	Project[i]=dict(myDict)
driver.close()
#################################################### PROFILE RATING ##############################################
if not Name:
	Profile_Completeness_Score=Profile_Completeness_Score+2
if not Position:
	Profile_Completeness_Score=Profile_Completeness_Score+2
if not Summary:
	Profile_Completeness_Score=Profile_Completeness_Score+2
if not Education:
	Profile_Completeness_Score=Profile_Completeness_Score+2
if not Number_of_Publications:
	Profile_Completeness_Score=Profile_Completeness_Score+2
if  bool(Publications):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Skills):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Experience):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Courses):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Headers):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Certifications):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Organisations):
	Profile_Completeness_Score=Profile_Completeness_Score+5
if  bool(Groups):
	Profile_Completeness_Score=Profile_Completeness_Score+5




Project_Score=(len(Headers)/len(Contributors))*10
if Project_Score>10:
	Project_Score=10

Publication_Score=len(Number_of_Publications)
if Publication_Score>5:
	Publication_Score=5

Skill_Score=len(Skills)/5
if Skill_Score>10:
	Skill_Score=10

Experience_Score=len(Experience)
if Experience_Score>5:
	Experience_Score=5

Certification_Score=(len(Courses)*len(Certifications))/5
if Certification_Score>10:
	Certification_Score=10

Organisational_Score=(len(Groups)+len(Organisations))/5
if Organisational_Score>10:
	Organisational_Score=10


Profile_Rating=Project_Score+Publication_Score+Skill_Score+Experience_Score+Certification_Score+Organisational_Score

Professional_Score=Profile_Rating+Profile_Completeness_Score




if Professional_Score<40:
	print("Well you need to improve. Your Professional_Score is:%d"%Professional_Score)
elif (Professional_Score>=40 and Professional_Score<60):
	print("You are good.Your Professional_Score is:%d" % Professional_Score)
elif (Professional_Score>=60 and Professional_Score<80):
	print("You Stand Average.Your Professional_Score is:%d"% Professional_Score)
elif (Professional_Score>=80 and Professional_Score<90):
	print("Near to perfect.Your Professional_Score is:%d"% Professional_Score)
else:
	print("Excellent.Your Professional_Score is:%d"% Professional_Score)