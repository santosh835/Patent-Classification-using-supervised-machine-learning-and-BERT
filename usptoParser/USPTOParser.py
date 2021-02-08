from bs4 import BeautifulSoup
import html
import pandas as pd
from collections import Counter


#This method will parse the xml saved in data folder
def parseUSPTOXML():

    # Load USPTO .xml document
    xml_text = html.unescape(open('data/2017/ipg161129/ipg161129.xml', 'r').read())
    
    #create empty dictionary
    tags = {"tags":[]}
    
    # Split out patent grants
    for patent in xml_text.split("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"):
      
        # Skip if it doesn't exist
        if patent is None or patent == "":
            continue 
        
        tag = {}
    
        # Load patent text as HTML document
        bs = BeautifulSoup(patent)                                     
        #serach for the grant
        application = bs.find('us-patent-grant')
        
        # Skip if it doesn't exist
        if application is None or application == "":
            continue 
        
        classifications = []
        for classes in bs.find_all('classifications-ipcr'):
            for el in classes.find_all('classification-ipcr'):
                classifications.append(el.find('section').text)
        
        if (len(classifications)==0):
            continue
        tag["classifications"]=Counter(classifications).most_common(1)[0][0]
    
        
        tag["patent_title"] = bs.find('invention-title').text
        tag["patent_num"] = bs.find('doc-number').text
        tag["patent_date"] = bs.find('publication-reference').find('date').text
        tag["application_type"] = bs.find('application-reference')['appl-type']
        
        inventors = []
        for parties in bs.find_all('inventor'):
            #for applicants in parties.find_all('applicants'):
                for el in parties.find_all('addressbook'):
                    inventors.append(el.find('first-name').text 
                                     + " " + el.find('last-name').text)
        tag["inventors"]=','.join(inventors)      
        abstracts = []
        for el in bs.find_all('abstract'):
            abstracts.append(el.text)
            
        descriptions = []
        for el in bs.find_all('description'):
            descriptions.append(el.text)
        
        claims = []
        for el in bs.find_all('claim'):
            claims.append(el.text)
        tag["abstracts"]=''.join(abstracts)   
        tag["description"]=''.join(descriptions)       
        tag["claims"]=''.join(claims)       
         
        tags["tags"]. append(tag)
    
    #create dataframe
    df = pd.DataFrame(tags["tags"])
    
    #save dataframe into csv format
    df.to_csv("PatentData.csv")
